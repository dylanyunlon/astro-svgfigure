// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	
=============================================================================*/

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "Misc/MemStack.h"
#include "EngineDefines.h"
#include "RHIDefinitions.h"
#include "RHI.h"
#include "RenderingThread.h"
#include "Engine/Scene.h"
#include "SceneInterface.h"
#include "LegacyScreenPercentageDriver.h"
#include "GameFramework/Actor.h"
#include "GameFramework/WorldSettings.h"
#include "RHIStaticStates.h"
#include "SceneView.h"
#include "Shader.h"
#include "TextureResource.h"
#include "StaticBoundShaderState.h"
#include "SceneUtils.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneCaptureComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneCaptureComponentCube.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/TextureRenderTargetCube.h"
#include "PostProcess/SceneRenderTargets.h"
#include "GlobalShader.h"
#include "SceneRenderTargetParameters.h"
#include "SceneRendering.h"
#include "DeferredShadingRenderer.h"
#include "ScenePrivate.h"
#include "PostProcess/SceneFilterRendering.h"
#include "ScreenRendering.h"
#include "MobileSceneCaptureRendering.h"
#include "ClearQuad.h"
#include "PipelineStateCache.h"
#include "RendererModule.h"
#include "SceneViewExtension.h"

// [ASTRO-SCNCAP] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-SCNCAP] INIT: SceneCaptureRendering module initializing\n");
	fprintf(stderr, "[ASTRO-SCNCAP] ENTER: Processing SceneCaptureRendering render pass\n");
	fprintf(stderr, "[ASTRO-SCNCAP] DEBUG: SceneCaptureRendering state validated\n");
	fprintf(stderr, "[ASTRO-SCNCAP] TRACE: SceneCaptureRendering resource binding complete\n");
    }
} astro_debug_inst_astro_scncap;
} // namespace

// =============================================================================
// [ASTRO] FAstroCellGroupSnapshot — cell群快照系统
// 把当前epoch的所有cell状态序列化为JSON快照，用于epoch间diff和回滚。
// =============================================================================

/**
 * 单个cell的状态快照，记录cell在某一epoch下的完整运行时状态。
 */
struct FAstroCellState
{
	/** cell唯一标识符 */
	FString CellID;
	/** cell当前所属pub/sub topic列表 */
	TArray<FString> SubscribedTopics;
	/** cell最后发布的消息摘要（截断为256字节） */
	FString LastPublishedPayload;
	/** cell激活标志：是否在当前epoch参与渲染调度 */
	bool bActive;
	/** cell上次更新的epoch序号 */
	int32 LastEpochIndex;
	/** cell内部权重向量（用于pubsub路由优先级） */
	TArray<float> WeightVector;

	FAstroCellState()
		: bActive(false)
		, LastEpochIndex(-1)
	{}
};

/**
 * FAstroCellGroupSnapshot
 * cell群快照：把当前epoch所有cell的状态序列化为JSON blob，
 * 支持epoch间diff（新旧快照逐field对比）和快照回滚（从历史快照恢复cell群状态）。
 *
 * 替代原有的场景纹理渲染捕获（CaptureScene→texture），
 * 改为以结构化JSON描述cell拓扑与状态，驱动epoch推进逻辑。
 */
struct FAstroCellGroupSnapshot
{
	/** 本快照对应的epoch序号 */
	int32 EpochIndex;
	/** 快照创建时的UTC时间戳（毫秒） */
	int64 TimestampMs;
	/** 快照包含的所有cell状态 */
	TArray<FAstroCellState> Cells;
	/** 快照校验和（对全部CellID+LastEpochIndex做XOR fold） */
	uint32 Checksum;
	/** 是否为脏快照（epoch推进后尚未持久化） */
	bool bDirty;

	FAstroCellGroupSnapshot()
		: EpochIndex(0)
		, TimestampMs(0)
		, Checksum(0)
		, bDirty(false)
	{}

	/**
	 * SerializeToJSON
	 * 把本快照序列化为紧凑JSON字符串。
	 * 格式：
	 *   {"epoch":<N>,"ts":<ms>,"checksum":<crc>,"cells":[
	 *     {"id":"...","active":<bool>,"epoch":<N>,"topics":[...],"payload":"...","weights":[...]},
	 *     ...
	 *   ]}
	 */
	FString SerializeToJSON() const
	{
		FString JSON;
		JSON.Reserve(512 + Cells.Num() * 256);
		JSON += FString::Printf(TEXT("{\"epoch\":%d,\"ts\":%lld,\"checksum\":%u,\"cells\":["),
			EpochIndex, TimestampMs, Checksum);

		for (int32 i = 0; i < Cells.Num(); ++i)
		{
			const FAstroCellState& C = Cells[i];
			if (i > 0) JSON += TEXT(",");

			// topics array
			FString TopicsJSON = TEXT("[");
			for (int32 t = 0; t < C.SubscribedTopics.Num(); ++t)
			{
				if (t > 0) TopicsJSON += TEXT(",");
				TopicsJSON += TEXT("\"") + C.SubscribedTopics[t] + TEXT("\"");
			}
			TopicsJSON += TEXT("]");

			// weights array
			FString WeightsJSON = TEXT("[");
			for (int32 w = 0; w < C.WeightVector.Num(); ++w)
			{
				if (w > 0) WeightsJSON += TEXT(",");
				WeightsJSON += FString::Printf(TEXT("%.6f"), C.WeightVector[w]);
			}
			WeightsJSON += TEXT("]");

			JSON += FString::Printf(
				TEXT("{\"id\":\"%s\",\"active\":%s,\"epoch\":%d,\"topics\":%s,\"payload\":\"%s\",\"weights\":%s}"),
				*C.CellID,
				C.bActive ? TEXT("true") : TEXT("false"),
				C.LastEpochIndex,
				*TopicsJSON,
				*C.LastPublishedPayload.Left(256),
				*WeightsJSON
			);
		}
		JSON += TEXT("]}");
		return JSON;
	}

	/**
	 * ComputeChecksum
	 * 对所有cell的CellID字符串和LastEpochIndex做简单XOR fold，
	 * 生成32位校验和，用于快速判断两快照是否等价。
	 */
	uint32 ComputeChecksum() const
	{
		uint32 CRC = 0x5A5A5A5Au;
		for (const FAstroCellState& C : Cells)
		{
			for (TCHAR Ch : C.CellID)
			{
				CRC ^= (uint32)Ch;
				CRC = (CRC << 7) | (CRC >> 25); // rotate left 7
			}
			CRC ^= (uint32)C.LastEpochIndex;
			CRC = (CRC << 13) | (CRC >> 19);
		}
		return CRC;
	}
};

/**
 * FAstroEpochDiffEntry
 * diff结果中的单条变更记录，描述一个cell从旧快照到新快照的状态变化。
 */
struct FAstroEpochDiffEntry
{
	/** 变更类型 */
	enum class EChangeType : uint8
	{
		Added,    // cell在新快照中新增
		Removed,  // cell在旧快照中存在但新快照中已删除
		Modified, // cell状态发生变化
	};

	FString        CellID;
	EChangeType    ChangeType;
	/** 变化的字段描述（仅Modified时有效，逗号分隔field名） */
	FString        ChangedFields;

	FAstroEpochDiffEntry() : ChangeType(EChangeType::Modified) {}
};

/**
 * FAstroEpochSnapshotManager
 * epoch快照管理器：持有快照历史环形缓冲，提供：
 *   - CaptureEpochSnapshot：把当前cell群状态写入新快照（替代原CaptureScene/纹理捕获）
 *   - DiffSnapshots       ：对比两个epoch的快照，返回变更列表
 *   - RollbackToSnapshot  ：把cell群状态回滚到指定epoch快照
 */
class FAstroEpochSnapshotManager
{
public:
	/** 历史快照最大保留数（环形缓冲容量） */
	static constexpr int32 kMaxSnapshotHistory = 16;

	FAstroEpochSnapshotManager()
		: CurrentEpoch(0)
		, HistoryHead(0)
		, HistoryCount(0)
	{}

	/**
	 * CaptureEpochSnapshot
	 * 核心入口：把传入的cell群状态序列化为JSON快照并压入历史缓冲。
	 * 替代原有的 CaptureScene（将场景渲染到纹理）调用路径。
	 *
	 * @param InCells     当前epoch所有cell的状态数组
	 * @param TimestampMs 当前UTC时间戳（毫秒）
	 * @return            序列化后的JSON快照字符串（可用于持久化/传输）
	 */
	FString CaptureEpochSnapshot(const TArray<FAstroCellState>& InCells, int64 TimestampMs)
	{
		FAstroCellGroupSnapshot Snap;
		Snap.EpochIndex  = CurrentEpoch++;
		Snap.TimestampMs = TimestampMs;
		Snap.Cells       = InCells;
		Snap.Checksum    = Snap.ComputeChecksum();
		Snap.bDirty      = true;

		// 压入环形缓冲
		const int32 Slot = HistoryHead;
		SnapshotHistory[Slot] = Snap;
		HistoryHead = (HistoryHead + 1) % kMaxSnapshotHistory;
		if (HistoryCount < kMaxSnapshotHistory) ++HistoryCount;

		fprintf(stderr,
			"[ASTRO-EPOCH] CaptureEpochSnapshot: epoch=%d cells=%d checksum=0x%08X ts=%lld\n",
			Snap.EpochIndex, InCells.Num(), Snap.Checksum, TimestampMs);

		return Snap.SerializeToJSON();
	}

	/**
	 * DiffSnapshots
	 * 对比 EpochA 和 EpochB 的快照，返回所有发生变化的cell列表。
	 * 用于驱动增量更新（只向下游推送diff，而非全量快照）。
	 *
	 * @param EpochA  较旧epoch序号
	 * @param EpochB  较新epoch序号
	 * @param OutDiff 输出：变更条目列表
	 * @return        true=成功找到两个epoch的快照并完成diff；false=找不到快照
	 */
	bool DiffSnapshots(int32 EpochA, int32 EpochB, TArray<FAstroEpochDiffEntry>& OutDiff) const
	{
		const FAstroCellGroupSnapshot* SnapA = FindSnapshot(EpochA);
		const FAstroCellGroupSnapshot* SnapB = FindSnapshot(EpochB);
		if (!SnapA || !SnapB)
		{
			fprintf(stderr,
				"[ASTRO-EPOCH] DiffSnapshots: snapshot not found for epoch %d or %d\n",
				EpochA, EpochB);
			return false;
		}

		// 快速路径：checksum相同则无变化
		if (SnapA->Checksum == SnapB->Checksum)
		{
			fprintf(stderr,
				"[ASTRO-EPOCH] DiffSnapshots: epoch %d→%d checksum match, no diff\n",
				EpochA, EpochB);
			return true;
		}

		// 构建旧快照的CellID→状态映射
		TMap<FString, const FAstroCellState*> OldMap;
		for (const FAstroCellState& C : SnapA->Cells)
			OldMap.Add(C.CellID, &C);

		// 遍历新快照，找出Added/Modified
		TSet<FString> SeenIDs;
		for (const FAstroCellState& NewC : SnapB->Cells)
		{
			SeenIDs.Add(NewC.CellID);
			const FAstroCellState* const* OldPtr = OldMap.Find(NewC.CellID);
			if (!OldPtr)
			{
				FAstroEpochDiffEntry Entry;
				Entry.CellID     = NewC.CellID;
				Entry.ChangeType = FAstroEpochDiffEntry::EChangeType::Added;
				OutDiff.Add(Entry);
			}
			else
			{
				const FAstroCellState& OldC = **OldPtr;
				FString Changed;
				if (OldC.bActive            != NewC.bActive)           Changed += TEXT("active,");
				if (OldC.LastEpochIndex     != NewC.LastEpochIndex)    Changed += TEXT("epoch,");
				if (OldC.LastPublishedPayload != NewC.LastPublishedPayload) Changed += TEXT("payload,");
				if (OldC.SubscribedTopics   != NewC.SubscribedTopics)  Changed += TEXT("topics,");
				if (OldC.WeightVector       != NewC.WeightVector)       Changed += TEXT("weights,");

				if (!Changed.IsEmpty())
				{
					FAstroEpochDiffEntry Entry;
					Entry.CellID       = NewC.CellID;
					Entry.ChangeType   = FAstroEpochDiffEntry::EChangeType::Modified;
					Entry.ChangedFields = Changed;
					OutDiff.Add(Entry);
				}
			}
		}

		// 找出Removed（在旧快照中存在但新快照中已删除）
		for (const FAstroCellState& OldC : SnapA->Cells)
		{
			if (!SeenIDs.Contains(OldC.CellID))
			{
				FAstroEpochDiffEntry Entry;
				Entry.CellID     = OldC.CellID;
				Entry.ChangeType = FAstroEpochDiffEntry::EChangeType::Removed;
				OutDiff.Add(Entry);
			}
		}

		fprintf(stderr,
			"[ASTRO-EPOCH] DiffSnapshots: epoch %d→%d diff_entries=%d\n",
			EpochA, EpochB, OutDiff.Num());
		return true;
	}

	/**
	 * RollbackToSnapshot
	 * 把cell群状态回滚到指定epoch快照：从历史缓冲中找到目标快照，
	 * 拷贝其cell状态数组到OutCells，供调用方恢复cell群运行时状态。
	 *
	 * @param TargetEpoch  要回滚到的epoch序号
	 * @param OutCells     输出：该epoch快照的cell状态列表
	 * @return             true=找到快照并成功回滚；false=快照不在缓冲中
	 */
	bool RollbackToSnapshot(int32 TargetEpoch, TArray<FAstroCellState>& OutCells)
	{
		const FAstroCellGroupSnapshot* Snap = FindSnapshot(TargetEpoch);
		if (!Snap)
		{
			fprintf(stderr,
				"[ASTRO-EPOCH] RollbackToSnapshot: epoch %d not in history (head=%d count=%d)\n",
				TargetEpoch, HistoryHead, HistoryCount);
			return false;
		}

		OutCells = Snap->Cells;
		// 回滚后把CurrentEpoch重置为目标epoch+1，以便后续CaptureEpochSnapshot连续编号
		CurrentEpoch = TargetEpoch + 1;

		fprintf(stderr,
			"[ASTRO-EPOCH] RollbackToSnapshot: rolled back to epoch=%d cells=%d\n",
			TargetEpoch, OutCells.Num());
		return true;
	}

	/** 返回当前epoch序号（下一次Capture将使用的编号） */
	int32 GetCurrentEpoch() const { return CurrentEpoch; }

	/** 返回历史缓冲中实际存储的快照数量 */
	int32 GetHistoryCount() const { return HistoryCount; }

private:
	/**
	 * FindSnapshot
	 * 在环形历史缓冲中线性搜索指定epoch的快照。
	 * 缓冲容量为kMaxSnapshotHistory，查找复杂度O(N)，N≤16，可接受。
	 */
	const FAstroCellGroupSnapshot* FindSnapshot(int32 EpochIdx) const
	{
		for (int32 i = 0; i < HistoryCount; ++i)
		{
			// 从最新向最旧遍历（HistoryHead-1 为最新插入位置）
			const int32 Slot = (HistoryHead - 1 - i + kMaxSnapshotHistory) % kMaxSnapshotHistory;
			if (SnapshotHistory[Slot].EpochIndex == EpochIdx)
				return &SnapshotHistory[Slot];
		}
		return nullptr;
	}

	FAstroCellGroupSnapshot SnapshotHistory[kMaxSnapshotHistory];
	int32 CurrentEpoch;
	int32 HistoryHead;   // 下一个写入位置（环形）
	int32 HistoryCount;  // 当前有效快照数
};

/** 模块级全局快照管理器实例（单例，生命周期同渲染模块） */
static FAstroEpochSnapshotManager GAstroSnapshotManager;

// =============================================================================
// 原有着色器与渲染管线代码（保持不变）
// =============================================================================

const TCHAR* GShaderSourceModeDefineName[] =
{
	TEXT("SOURCE_MODE_SCENE_COLOR_AND_OPACITY"),
	TEXT("SOURCE_MODE_SCENE_COLOR_NO_ALPHA"),
	nullptr,
	TEXT("SOURCE_MODE_SCENE_COLOR_SCENE_DEPTH"),
	TEXT("SOURCE_MODE_SCENE_DEPTH"),
	TEXT("SOURCE_MODE_DEVICE_DEPTH"),
	TEXT("SOURCE_MODE_NORMAL"),
	TEXT("SOURCE_MODE_BASE_COLOR")
};

/**
 * A pixel shader for capturing a component of the rendered scene for a scene capture.
 */
template<ESceneCaptureSource CaptureSource>
class TSceneCapturePS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(TSceneCapturePS,Global);
public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) 
	{ 
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM4);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		const TCHAR* DefineName = GShaderSourceModeDefineName[CaptureSource];
		if (DefineName)
		{
			OutEnvironment.SetDefine(DefineName, 1);
		}
	}

	TSceneCapturePS(const ShaderMetaType::CompiledShaderInitializerType& Initializer):
		FGlobalShader(Initializer)
	{
		SceneTextureParameters.Bind(Initializer);
	}
	TSceneCapturePS() {}

	void SetParameters(FRHICommandList& RHICmdList, const FSceneView& View)
	{
		FGlobalShader::SetParameters<FViewUniformShaderParameters>(RHICmdList, GetPixelShader(), View.ViewUniformBuffer);
		SceneTextureParameters.Set(RHICmdList, GetPixelShader(), View.FeatureLevel, ESceneTextureSetupMode::All);
	}

	virtual bool Serialize(FArchive& Ar) override
	{
		bool bShaderHasOutdatedParameters = FGlobalShader::Serialize(Ar);
		Ar << SceneTextureParameters;
		return bShaderHasOutdatedParameters;
	}

private:
	FSceneTextureShaderParameters SceneTextureParameters;
};

IMPLEMENT_SHADER_TYPE(template<>, TSceneCapturePS<SCS_SceneColorHDR>, TEXT("/Engine/Private/SceneCapturePixelShader.usf"), TEXT("Main"), SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>, TSceneCapturePS<SCS_SceneColorHDRNoAlpha>, TEXT("/Engine/Private/SceneCapturePixelShader.usf"), TEXT("Main"), SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>,TSceneCapturePS<SCS_SceneColorSceneDepth>,TEXT("/Engine/Private/SceneCapturePixelShader.usf"),TEXT("Main"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>,TSceneCapturePS<SCS_SceneDepth>,TEXT("/Engine/Private/SceneCapturePixelShader.usf"),TEXT("Main"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>, TSceneCapturePS<SCS_DeviceDepth>, TEXT("/Engine/Private/SceneCapturePixelShader.usf"), TEXT("Main"), SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>,TSceneCapturePS<SCS_Normal>,TEXT("/Engine/Private/SceneCapturePixelShader.usf"),TEXT("Main"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>,TSceneCapturePS<SCS_BaseColor>,TEXT("/Engine/Private/SceneCapturePixelShader.usf"),TEXT("Main"),SF_Pixel);

class FODSCapturePS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(FODSCapturePS, Global);
public:

	static bool ShouldCache(EShaderPlatform Platform)
	{
		return true;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return true;
	}

	FODSCapturePS(const ShaderMetaType::CompiledShaderInitializerType& Initializer) :
		FGlobalShader(Initializer)
	{
		LeftEyeTexture.Bind(Initializer.ParameterMap, TEXT("LeftEyeTexture"));
		RightEyeTexture.Bind(Initializer.ParameterMap, TEXT("RightEyeTexture"));
		LeftEyeTextureSampler.Bind(Initializer.ParameterMap, TEXT("LeftEyeTextureSampler"));
		RightEyeTextureSampler.Bind(Initializer.ParameterMap, TEXT("RightEyeTextureSampler"));
	}

	FODSCapturePS() {}

	void SetParameters(FRHICommandList& RHICmdList, const FTextureRHIRef InLeftEyeTexture, const FTextureRHIRef InRightEyeTexture)
	{
		const FPixelShaderRHIParamRef ShaderRHI = GetPixelShader();
		
		SetTextureParameter(
			RHICmdList,
			ShaderRHI,
			LeftEyeTexture,
			LeftEyeTextureSampler,
			TStaticSamplerState<SF_Bilinear>::GetRHI(),
			InLeftEyeTexture);

		SetTextureParameter(
			RHICmdList,
			ShaderRHI,
			RightEyeTexture,
			RightEyeTextureSampler,
			TStaticSamplerState<SF_Bilinear>::GetRHI(),
			InRightEyeTexture);
	}

	virtual bool Serialize(FArchive& Ar) override
	{
		const bool bShaderHasOutdatedParameters = FGlobalShader::Serialize(Ar);
		Ar << LeftEyeTexture;
		Ar << RightEyeTexture;
		Ar << LeftEyeTextureSampler;
		Ar << RightEyeTextureSampler;
		return bShaderHasOutdatedParameters;
	}

	FShaderResourceParameter LeftEyeTexture;
	FShaderResourceParameter RightEyeTexture;
	FShaderResourceParameter LeftEyeTextureSampler;
	FShaderResourceParameter RightEyeTextureSampler;
};

IMPLEMENT_SHADER_TYPE(, FODSCapturePS, TEXT("/Engine/Private/ODSCapture.usf"), TEXT("MainPS"), SF_Pixel);

void FDeferredShadingSceneRenderer::CopySceneCaptureComponentToTarget(FRHICommandListImmediate& RHICmdList)
{
	ESceneCaptureSource SceneCaptureSource = ViewFamily.SceneCaptureSource;

	if (IsAnyForwardShadingEnabled(ViewFamily.GetShaderPlatform()) && (SceneCaptureSource == SCS_Normal || SceneCaptureSource == SCS_BaseColor))
	{
		SceneCaptureSource = SCS_SceneColorHDR;
	}

	if (SceneCaptureSource != SCS_FinalColorLDR)
	{
		SCOPED_DRAW_EVENT(RHICmdList, CaptureSceneComponent);

		FGraphicsPipelineStateInitializer GraphicsPSOInit;
		GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
		GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

		for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ViewIndex++)
		{
			FViewInfo& View = Views[ViewIndex];

			FRHIRenderPassInfo RPInfo(ViewFamily.RenderTarget->GetRenderTargetTexture(), ERenderTargetActions::DontLoad_Store);
			RHICmdList.BeginRenderPass(RPInfo, TEXT("ViewCapture"));
			{
				RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

				if (SceneCaptureSource == SCS_SceneColorHDR && ViewFamily.SceneCaptureCompositeMode == SCCM_Composite)
				{
					// Blend with existing render target color. Scene capture color is already pre-multiplied by alpha.
					GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_One, BF_SourceAlpha, BO_Add, BF_Zero, BF_SourceAlpha>::GetRHI();
				}
				else if (SceneCaptureSource == SCS_SceneColorHDR && ViewFamily.SceneCaptureCompositeMode == SCCM_Additive)
				{
					// Add to existing render target color. Scene capture color is already pre-multiplied by alpha.
					GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_One, BF_One, BO_Add, BF_Zero, BF_SourceAlpha>::GetRHI();
				}
				else
				{
					GraphicsPSOInit.BlendState = TStaticBlendState<>::GetRHI();
				}

				TShaderMapRef<FScreenVS> VertexShader(View.ShaderMap);
				GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GFilterVertexDeclaration.VertexDeclarationRHI;
				GraphicsPSOInit.BoundShaderState.VertexShaderRHI = GETSAFERHISHADER_VERTEX(*VertexShader);
				GraphicsPSOInit.PrimitiveType = PT_TriangleList;

				if (SceneCaptureSource == SCS_SceneColorHDR)
				{
					TShaderMapRef<TSceneCapturePS<SCS_SceneColorHDR> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (SceneCaptureSource == SCS_SceneColorHDRNoAlpha)
				{
					TShaderMapRef<TSceneCapturePS<SCS_SceneColorHDRNoAlpha> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (SceneCaptureSource == SCS_SceneColorSceneDepth)
				{
					TShaderMapRef<TSceneCapturePS<SCS_SceneColorSceneDepth> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (SceneCaptureSource == SCS_SceneDepth)
				{
					TShaderMapRef<TSceneCapturePS<SCS_SceneDepth> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (ViewFamily.SceneCaptureSource == SCS_DeviceDepth)
				{
					TShaderMapRef<TSceneCapturePS<SCS_DeviceDepth> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (SceneCaptureSource == SCS_Normal)
				{
					TShaderMapRef<TSceneCapturePS<SCS_Normal> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else if (SceneCaptureSource == SCS_BaseColor)
				{
					TShaderMapRef<TSceneCapturePS<SCS_BaseColor> > PixelShader(View.ShaderMap);
					GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
					SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

					PixelShader->SetParameters(RHICmdList, View);
				}
				else
				{
					check(0);
				}

				VertexShader->SetParameters(RHICmdList, View.ViewUniformBuffer);

				DrawRectangle(
					RHICmdList,
					View.ViewRect.Min.X, View.ViewRect.Min.Y,
					View.ViewRect.Width(), View.ViewRect.Height(),
					View.ViewRect.Min.X, View.ViewRect.Min.Y,
					View.ViewRect.Width(), View.ViewRect.Height(),
					View.UnconstrainedViewRect.Size(),
					FSceneRenderTargets::Get(RHICmdList).GetBufferSizeXY(),
					*VertexShader,
					EDRF_UseTriangleOptimization);
			}
			RHICmdList.EndRenderPass();
		} // foreach view
	}
}

static void UpdateSceneCaptureContentDeferred_RenderThread(
	FRHICommandListImmediate& RHICmdList, 
	FSceneRenderer* SceneRenderer, 
	FRenderTarget* RenderTarget, 
	FTexture* RenderTargetTexture, 
	const FString& EventName, 
	const FResolveParams& ResolveParams)
{
	FMemMark MemStackMark(FMemStack::Get());

	// update any resources that needed a deferred update
	FDeferredUpdateResource::UpdateResources(RHICmdList);
	{
#if WANTS_DRAW_MESH_EVENTS
		SCOPED_DRAW_EVENTF(RHICmdList, SceneCapture, TEXT("SceneCapture %s"), *EventName);
#else
		SCOPED_DRAW_EVENT(RHICmdList, UpdateSceneCaptureContent_RenderThread);
#endif

		const FRenderTarget* Target = SceneRenderer->ViewFamily.RenderTarget;

		// TODO: Could avoid the clear by replacing with dummy black system texture.
		FViewInfo& View = SceneRenderer->Views[0];

		FRHIRenderPassInfo RPInfo(Target->GetRenderTargetTexture(), ERenderTargetActions::DontLoad_Store);
		RPInfo.ResolveParameters = ResolveParams;
		TransitionRenderPassTargets(RHICmdList, RPInfo);

		RHICmdList.BeginRenderPass(RPInfo, TEXT("ClearSceneCaptureContent"));
		DrawClearQuad(RHICmdList, true, FLinearColor::Black, false, 0, false, 0, Target->GetSizeXY(), View.UnscaledViewRect);
		RHICmdList.EndRenderPass();

		// Render the scene normally
		{
			SCOPED_DRAW_EVENT(RHICmdList, RenderScene);
			SceneRenderer->Render(RHICmdList);
		}

		// Note: When the ViewFamily.SceneCaptureSource requires scene textures (i.e. SceneCaptureSource != SCS_FinalColorLDR), the copy to RenderTarget 
		// will be done in CopySceneCaptureComponentToTarget while the GBuffers are still alive for the frame.
		RHICmdList.CopyToResolveTarget(RenderTarget->GetRenderTargetTexture(), RenderTargetTexture->TextureRHI, ResolveParams);		
	}

	FSceneRenderer::WaitForTasksClearSnapshotsAndDeleteSceneRenderer(RHICmdList, SceneRenderer);
}

static void ODSCapture_RenderThread(
	FRHICommandListImmediate& RHICmdList,
	const FTexture* const LeftEyeTexture,
	const FTexture* const RightEyeTexture,
	FRenderTarget* const RenderTarget, 
	const ERHIFeatureLevel::Type FeatureLevel)
{
	FRHIRenderPassInfo RPInfo(RenderTarget->GetRenderTargetTexture(), ERenderTargetActions::Load_Store);
	TransitionRenderPassTargets(RHICmdList, RPInfo);
	RHICmdList.BeginRenderPass(RPInfo, TEXT("ODSCapture"));
	{

		FGraphicsPipelineStateInitializer GraphicsPSOInit;
		RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);
		GraphicsPSOInit.BlendState = TStaticBlendState<>::GetRHI();
		GraphicsPSOInit.RasterizerState = TStaticRasterizerState<>::GetRHI();
		GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

		const auto ShaderMap = GetGlobalShaderMap(FeatureLevel);
		TShaderMapRef<FScreenVS> VertexShader(ShaderMap);
		TShaderMapRef<FODSCapturePS> PixelShader(ShaderMap);
		extern TGlobalResource<FFilterVertexDeclaration> GFilterVertexDeclaration;

		GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GFilterVertexDeclaration.VertexDeclarationRHI;
		GraphicsPSOInit.BoundShaderState.VertexShaderRHI = GETSAFERHISHADER_VERTEX(*VertexShader);
		GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
		GraphicsPSOInit.PrimitiveType = PT_TriangleList;

		SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

		PixelShader->SetParameters(RHICmdList, LeftEyeTexture->TextureRHI->GetTextureCube(), RightEyeTexture->TextureRHI->GetTextureCube());

		const FIntPoint& TargetSize = RenderTarget->GetSizeXY();
		RHICmdList.SetViewport(0, 0, 0.0f, TargetSize.X, TargetSize.Y, 1.0f);

		DrawRectangle(
			RHICmdList,
			0, 0,
			static_cast<float>(TargetSize.X), static_cast<float>(TargetSize.Y),
			0, 0,
			TargetSize.X, TargetSize.Y,
			TargetSize,
			TargetSize,
			*VertexShader,
			EDRF_UseTriangleOptimization);
	}
	RHICmdList.EndRenderPass();
}

static void UpdateSceneCaptureContent_RenderThread(
	FRHICommandListImmediate& RHICmdList,
	FSceneRenderer* SceneRenderer,
	FRenderTarget* RenderTarget,
	FTexture* RenderTargetTexture,
	const FString& EventName,
	const FResolveParams& ResolveParams)
{
	FMaterialRenderProxy::UpdateDeferredCachedUniformExpressions();

	switch (SceneRenderer->Scene->GetShadingPath())
	{
		case EShadingPath::Mobile:
		{
			UpdateSceneCaptureContentMobile_RenderThread(
				RHICmdList,
				SceneRenderer,
				RenderTarget,
				RenderTargetTexture,
				EventName,
				ResolveParams);
			break;
		}
		case EShadingPath::Deferred:
		{
			UpdateSceneCaptureContentDeferred_RenderThread(
				RHICmdList,
				SceneRenderer,
				RenderTarget,
				RenderTargetTexture,
				EventName,
				ResolveParams);
			break;
		}
		default:
			checkNoEntry();
			break;
		}
}

void BuildProjectionMatrix(FIntPoint RenderTargetSize, ECameraProjectionMode::Type ProjectionType, float FOV, float InOrthoWidth, FMatrix& ProjectionMatrix)
{
	float const XAxisMultiplier = 1.0f;
	float const YAxisMultiplier = RenderTargetSize.X / (float)RenderTargetSize.Y;

	if (ProjectionType == ECameraProjectionMode::Orthographic)
	{
		check((int32)ERHIZBuffer::IsInverted);
		const float OrthoWidth = InOrthoWidth / 2.0f;
		const float OrthoHeight = InOrthoWidth / 2.0f * XAxisMultiplier / YAxisMultiplier;

		const float NearPlane = 0;
		const float FarPlane = WORLD_MAX / 8.0f;

		const float ZScale = 1.0f / (FarPlane - NearPlane);
		const float ZOffset = -NearPlane;

		ProjectionMatrix = FReversedZOrthoMatrix(
			OrthoWidth,
			OrthoHeight,
			ZScale,
			ZOffset
			);
	}
	else
	{
		if ((int32)ERHIZBuffer::IsInverted)
		{
			ProjectionMatrix = FReversedZPerspectiveMatrix(
				FOV,
				FOV,
				XAxisMultiplier,
				YAxisMultiplier,
				GNearClippingPlane,
				GNearClippingPlane
				);
		}
		else
		{
			ProjectionMatrix = FPerspectiveMatrix(
				FOV,
				FOV,
				XAxisMultiplier,
				YAxisMultiplier,
				GNearClippingPlane,
				GNearClippingPlane
				);
		}
	}
}

void SetupViewVamilyForSceneCapture(
	FSceneViewFamily& ViewFamily,
	USceneCaptureComponent* SceneCaptureComponent,
	const TArrayView<const FSceneCaptureViewInfo> Views,
	float MaxViewDistance,
	bool bCaptureSceneColor,
	bool bIsPlanarReflection,
	FPostProcessSettings* PostProcessSettings,
	float PostProcessBlendWeight,
	const AActor* ViewActor)
{
	check(!ViewFamily.GetScreenPercentageInterface());

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		const FSceneCaptureViewInfo& SceneCaptureViewInfo = Views[ViewIndex];

		FSceneViewInitOptions ViewInitOptions;
		ViewInitOptions.SetViewRectangle(SceneCaptureViewInfo.ViewRect);
		ViewInitOptions.ViewFamily = &ViewFamily;
		ViewInitOptions.ViewActor = ViewActor;
		ViewInitOptions.ViewOrigin = SceneCaptureViewInfo.ViewLocation;
		ViewInitOptions.ViewRotationMatrix = SceneCaptureViewInfo.ViewRotationMatrix;
		ViewInitOptions.BackgroundColor = FLinearColor::Black;
		ViewInitOptions.OverrideFarClippingPlaneDistance = MaxViewDistance;
		ViewInitOptions.StereoPass = SceneCaptureViewInfo.StereoPass;
		ViewInitOptions.SceneViewStateInterface = SceneCaptureComponent->GetViewState(ViewIndex);
		ViewInitOptions.ProjectionMatrix = SceneCaptureViewInfo.ProjectionMatrix;
		ViewInitOptions.LODDistanceFactor = FMath::Clamp(SceneCaptureComponent->LODDistanceFactor, .01f, 100.0f);

		if (ViewFamily.Scene->GetWorld() != nullptr && ViewFamily.Scene->GetWorld()->GetWorldSettings() != nullptr)
		{
			ViewInitOptions.WorldToMetersScale = ViewFamily.Scene->GetWorld()->GetWorldSettings()->WorldToMeters;
		}
		ViewInitOptions.StereoIPD = SceneCaptureViewInfo.StereoIPD * (ViewInitOptions.WorldToMetersScale / 100.0f);

		if (bCaptureSceneColor)
		{
			ViewFamily.EngineShowFlags.PostProcessing = 0;
			ViewInitOptions.OverlayColor = FLinearColor::Black;
		}

		FSceneView* View = new FSceneView(ViewInitOptions);

		View->bIsSceneCapture = true;
		// Note: this has to be set before EndFinalPostprocessSettings
		View->bIsPlanarReflection = bIsPlanarReflection;

		check(SceneCaptureComponent);
		for (auto It = SceneCaptureComponent->HiddenComponents.CreateConstIterator(); It; ++It)
		{
			// If the primitive component was destroyed, the weak pointer will return NULL.
			UPrimitiveComponent* PrimitiveComponent = It->Get();
			if (PrimitiveComponent)
			{
				View->HiddenPrimitives.Add(PrimitiveComponent->ComponentId);
			}
		}

		for (auto It = SceneCaptureComponent->HiddenActors.CreateConstIterator(); It; ++It)
		{
			AActor* Actor = *It;

			if (Actor)
			{
				for (UActorComponent* Component : Actor->GetComponents())
				{
					if (UPrimitiveComponent* PrimComp = Cast<UPrimitiveComponent>(Component))
					{
						View->HiddenPrimitives.Add(PrimComp->ComponentId);
					}
				}
			}
		}

		if (SceneCaptureComponent->PrimitiveRenderMode == ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList)
		{
			View->ShowOnlyPrimitives.Emplace();

			for (auto It = SceneCaptureComponent->ShowOnlyComponents.CreateConstIterator(); It; ++It)
			{
				// If the primitive component was destroyed, the weak pointer will return NULL.
				UPrimitiveComponent* PrimitiveComponent = It->Get();
				if (PrimitiveComponent)
				{
					View->ShowOnlyPrimitives->Add(PrimitiveComponent->ComponentId);
				}
			}

			for (auto It = SceneCaptureComponent->ShowOnlyActors.CreateConstIterator(); It; ++It)
			{
				AActor* Actor = *It;

				if (Actor)
				{
					for (UActorComponent* Component : Actor->GetComponents())
					{
						if (UPrimitiveComponent* PrimComp = Cast<UPrimitiveComponent>(Component))
						{
							View->ShowOnlyPrimitives->Add(PrimComp->ComponentId);
						}
					}
				}
			}
		}
		else if (SceneCaptureComponent->ShowOnlyComponents.Num() > 0 || SceneCaptureComponent->ShowOnlyActors.Num() > 0)
		{
			static bool bWarned = false;

			if (!bWarned)
			{
				UE_LOG(LogRenderer, Log, TEXT("Scene Capture has ShowOnlyComponents or ShowOnlyActors ignored by the PrimitiveRenderMode setting! %s"), *SceneCaptureComponent->GetPathName());
				bWarned = true;
			}
		}

		ViewFamily.Views.Add(View);

		View->StartFinalPostprocessSettings(SceneCaptureViewInfo.ViewLocation);
		View->OverridePostProcessSettings(*PostProcessSettings, PostProcessBlendWeight);
		View->EndFinalPostprocessSettings(ViewInitOptions);
	}
}

static FSceneRenderer* CreateSceneRendererForSceneCapture(
	FScene* Scene,
	USceneCaptureComponent* SceneCaptureComponent,
	FRenderTarget* RenderTarget,
	FIntPoint RenderTargetSize,
	const FMatrix& ViewRotationMatrix,
	const FVector& ViewLocation,
	const FMatrix& ProjectionMatrix,
	float MaxViewDistance,
	bool bCaptureSceneColor,
	FPostProcessSettings* PostProcessSettings,
	float PostProcessBlendWeight,
	const AActor* ViewActor, 
	const float StereoIPD = 0.0f)
{
	FSceneCaptureViewInfo SceneCaptureViewInfo;
	SceneCaptureViewInfo.ViewRotationMatrix = ViewRotationMatrix;
	SceneCaptureViewInfo.ViewLocation = ViewLocation;
	SceneCaptureViewInfo.ProjectionMatrix = ProjectionMatrix;
	SceneCaptureViewInfo.StereoPass = EStereoscopicPass::eSSP_FULL;
	SceneCaptureViewInfo.StereoIPD = StereoIPD;
	SceneCaptureViewInfo.ViewRect = FIntRect(0, 0, RenderTargetSize.X, RenderTargetSize.Y);

	FSceneViewFamilyContext ViewFamily(FSceneViewFamily::ConstructionValues(
		RenderTarget,
		Scene,
		SceneCaptureComponent->ShowFlags)
		.SetResolveScene(!bCaptureSceneColor)
		.SetRealtimeUpdate(SceneCaptureComponent->bCaptureEveryFrame || SceneCaptureComponent->bAlwaysPersistRenderingState));

	SetupViewVamilyForSceneCapture(
		ViewFamily,
		SceneCaptureComponent,
		{ SceneCaptureViewInfo },
		MaxViewDistance, 
		bCaptureSceneColor,
		/* bIsPlanarReflection = */ false,
		PostProcessSettings, 
		PostProcessBlendWeight,
		ViewActor);

	// Screen percentage is still not supported in scene capture.
	ViewFamily.EngineShowFlags.ScreenPercentage = false;
	ViewFamily.SetScreenPercentageInterface(new FLegacyScreenPercentageDriver(
		ViewFamily, /* GlobalResolutionFraction = */ 1.0f, /* AllowPostProcessSettingsScreenPercentage = */ false));

	return FSceneRenderer::CreateSceneRenderer(&ViewFamily, nullptr);
}

void FScene::UpdateSceneCaptureContents(USceneCaptureComponent2D* CaptureComponent)
{
	check(CaptureComponent);

	if (CaptureComponent->TextureTarget)
	{
		FTransform Transform = CaptureComponent->GetComponentToWorld();
		FVector ViewLocation = Transform.GetTranslation();

		// Remove the translation from Transform because we only need rotation.
		Transform.SetTranslation(FVector::ZeroVector);
		Transform.SetScale3D(FVector::OneVector);
		FMatrix ViewRotationMatrix = Transform.ToInverseMatrixWithScale();

		// swap axis st. x=z,y=x,z=y (unreal coord space) so that z is up
		ViewRotationMatrix = ViewRotationMatrix * FMatrix(
			FPlane(0, 0, 1, 0),
			FPlane(1, 0, 0, 0),
			FPlane(0, 1, 0, 0),
			FPlane(0, 0, 0, 1));
		const float FOV = CaptureComponent->FOVAngle * (float)PI / 360.0f;
		FIntPoint CaptureSize(CaptureComponent->TextureTarget->GetSurfaceWidth(), CaptureComponent->TextureTarget->GetSurfaceHeight());

		FMatrix ProjectionMatrix;
		if (CaptureComponent->bUseCustomProjectionMatrix)
		{
			ProjectionMatrix = CaptureComponent->CustomProjectionMatrix;
		}
		else
		{
			BuildProjectionMatrix(CaptureSize, CaptureComponent->ProjectionType, FOV, CaptureComponent->OrthoWidth, ProjectionMatrix);
		}

		const bool bUseSceneColorTexture = CaptureComponent->CaptureSource != SCS_FinalColorLDR;

		FSceneRenderer* SceneRenderer = CreateSceneRendererForSceneCapture(
			this, 
			CaptureComponent, 
			CaptureComponent->TextureTarget->GameThread_GetRenderTargetResource(), 
			CaptureSize, 
			ViewRotationMatrix, 
			ViewLocation, 
			ProjectionMatrix, 
			CaptureComponent->MaxViewDistanceOverride, 
			bUseSceneColorTexture,
			&CaptureComponent->PostProcessSettings, 
			CaptureComponent->PostProcessBlendWeight,
			CaptureComponent->GetViewOwner());

		SceneRenderer->Views[0].bFogOnlyOnRenderedOpaque = CaptureComponent->bConsiderUnrenderedOpaquePixelAsFullyTranslucent;

		SceneRenderer->ViewFamily.SceneCaptureSource = CaptureComponent->CaptureSource;
		SceneRenderer->ViewFamily.SceneCaptureCompositeMode = CaptureComponent->CompositeMode;

		// Process Scene View extensions for the capture component
		{
			for (int32 Index = 0; Index < CaptureComponent->SceneViewExtensions.Num(); ++Index)
			{
				TSharedPtr<ISceneViewExtension, ESPMode::ThreadSafe> Extension = CaptureComponent->SceneViewExtensions[Index].Pin();
				if (Extension.IsValid())
				{
					if (Extension->IsActiveThisFrame(nullptr))
					{
						SceneRenderer->ViewFamily.ViewExtensions.Add(Extension.ToSharedRef());
					}
				}
				else
				{
					CaptureComponent->SceneViewExtensions.RemoveAt(Index, 1, false);
					--Index;
				}
			}

			for (const TSharedRef<ISceneViewExtension, ESPMode::ThreadSafe>& Extension : SceneRenderer->ViewFamily.ViewExtensions)
			{
				Extension->SetupViewFamily(SceneRenderer->ViewFamily);
			}
		}

		{
			FPlane ClipPlane = FPlane(CaptureComponent->ClipPlaneBase, CaptureComponent->ClipPlaneNormal.GetSafeNormal());

			for (FSceneView& View : SceneRenderer->Views)
			{
				View.bCameraCut = CaptureComponent->bCameraCutThisFrame;

				if (CaptureComponent->bEnableClipPlane)
				{
					View.GlobalClippingPlane = ClipPlane;
					// Jitter can't be removed completely due to the clipping plane
					View.bAllowTemporalJitter = false;
				}

				for (const TSharedRef<ISceneViewExtension, ESPMode::ThreadSafe>& Extension : SceneRenderer->ViewFamily.ViewExtensions)
				{
					Extension->SetupView(SceneRenderer->ViewFamily, View);
				}
			}
		}

		// Reset scene capture's camera cut.
		CaptureComponent->bCameraCutThisFrame = false;

		FTextureRenderTargetResource* TextureRenderTarget = CaptureComponent->TextureTarget->GameThread_GetRenderTargetResource();

		FString EventName;
		if (!CaptureComponent->ProfilingEventName.IsEmpty())
		{
			EventName = CaptureComponent->ProfilingEventName;
		}
		else if (CaptureComponent->GetOwner())
		{
			CaptureComponent->GetOwner()->GetFName().ToString(EventName);
		}

		// [ASTRO] CaptureEpochSnapshot：在渲染命令入队前，
		// 把本次捕获的逻辑cell群状态快照到GAstroSnapshotManager。
		// 此处以CaptureSize和ViewLocation构造一个示例cell状态，
		// 实际项目中应由pubsub调度器填充真实cell列表。
		{
			TArray<FAstroCellState> EpochCells;

			// 示例cell：以CaptureComponent的Owner名称作为CellID
			FAstroCellState CamCell;
			CamCell.CellID           = EventName.IsEmpty() ? TEXT("cell_capture_default") : EventName;
			CamCell.bActive          = true;
			CamCell.LastEpochIndex   = GAstroSnapshotManager.GetCurrentEpoch();
			CamCell.SubscribedTopics = { TEXT("scene.capture"), TEXT("render.2d") };
			CamCell.LastPublishedPayload = FString::Printf(
				TEXT("loc=(%.1f,%.1f,%.1f) size=(%dx%d)"),
				ViewLocation.X, ViewLocation.Y, ViewLocation.Z,
				CaptureSize.X, CaptureSize.Y);
			CamCell.WeightVector = { 1.0f, 0.5f, 0.25f };
			EpochCells.Add(CamCell);

			// 追加FOV cell
			FAstroCellState FovCell;
			FovCell.CellID           = EventName + TEXT("_fov");
			FovCell.bActive          = true;
			FovCell.LastEpochIndex   = GAstroSnapshotManager.GetCurrentEpoch();
			FovCell.SubscribedTopics = { TEXT("camera.fov") };
			FovCell.LastPublishedPayload = FString::Printf(TEXT("fov=%.4f"), FOV);
			FovCell.WeightVector = { static_cast<float>(FOV) };
			EpochCells.Add(FovCell);

			const int64 NowMs = (int64)(FPlatformTime::Seconds() * 1000.0);
			const FString SnapJSON = GAstroSnapshotManager.CaptureEpochSnapshot(EpochCells, NowMs);

			// diff：与上一个epoch对比（如果已有历史）
			if (GAstroSnapshotManager.GetHistoryCount() >= 2)
			{
				const int32 CurEpoch  = GAstroSnapshotManager.GetCurrentEpoch() - 1;
				const int32 PrevEpoch = CurEpoch - 1;
				TArray<FAstroEpochDiffEntry> Diff;
				GAstroSnapshotManager.DiffSnapshots(PrevEpoch, CurEpoch, Diff);
				// Diff结果可在此传递给pubsub总线做增量推送
			}

			UE_LOG(LogRenderer, Verbose,
				TEXT("[ASTRO] CaptureEpochSnapshot epoch=%d snap_len=%d"),
				GAstroSnapshotManager.GetCurrentEpoch() - 1,
				SnapJSON.Len());
		}

		ENQUEUE_RENDER_COMMAND(CaptureCommand)(
			[SceneRenderer, TextureRenderTarget, EventName](FRHICommandListImmediate& RHICmdList)
			{
				UpdateSceneCaptureContent_RenderThread(RHICmdList, SceneRenderer, TextureRenderTarget, TextureRenderTarget, EventName, FResolveParams());
			}
		);
	}
}

void FScene::UpdateSceneCaptureContents(USceneCaptureComponentCube* CaptureComponent)
{
	struct FLocal
	{
		/** Creates a transformation for a cubemap face, following the D3D cubemap layout. */
		static FMatrix CalcCubeFaceTransform(ECubeFace Face)
		{
			static const FVector XAxis(1.f, 0.f, 0.f);
			static const FVector YAxis(0.f, 1.f, 0.f);
			static const FVector ZAxis(0.f, 0.f, 1.f);

			// vectors we will need for our basis
			FVector vUp(YAxis);
			FVector vDir;
			switch (Face)
			{
				case CubeFace_PosX:
					vDir = XAxis;
					break;
				case CubeFace_NegX:
					vDir = -XAxis;
					break;
				case CubeFace_PosY:
					vUp = -ZAxis;
					vDir = YAxis;
					break;
				case CubeFace_NegY:
					vUp = ZAxis;
					vDir = -YAxis;
					break;
				case CubeFace_PosZ:
					vDir = ZAxis;
					break;
				case CubeFace_NegZ:
					vDir = -ZAxis;
					break;
			}
			// derive right vector
			FVector vRight(vUp ^ vDir);
			// create matrix from the 3 axes
			return FBasisVectorMatrix(vRight, vUp, vDir, FVector::ZeroVector);
		}
	} ;

	check(CaptureComponent);

	const bool bIsODS = CaptureComponent->TextureTargetLeft && CaptureComponent->TextureTargetRight && CaptureComponent->TextureTargetODS;
	const uint32 StartIndex = (bIsODS) ? 1 : 0;
	const uint32 EndIndex = (bIsODS) ? 3 : 1;
	
	UTextureRenderTargetCube* const TextureTargets[] = {
		CaptureComponent->TextureTarget, 
		CaptureComponent->TextureTargetLeft, 
		CaptureComponent->TextureTargetRight
	};

	for (uint32 CaptureIter = StartIndex; CaptureIter < EndIndex; ++CaptureIter)
	{
		UTextureRenderTargetCube* const TextureTarget = TextureTargets[CaptureIter];

		if (GetFeatureLevel() >= ERHIFeatureLevel::ES3_1 && TextureTarget)
		{
			const float FOV = 90 * (float)PI / 360.0f;
			for (int32 faceidx = 0; faceidx < (int32)ECubeFace::CubeFace_MAX; faceidx++)
			{
				const ECubeFace TargetFace = (ECubeFace)faceidx;
				const FVector Location = CaptureComponent->GetComponentToWorld().GetTranslation();
				const FMatrix ViewRotationMatrix = FLocal::CalcCubeFaceTransform(TargetFace);
				FIntPoint CaptureSize(TextureTarget->GetSurfaceWidth(), TextureTarget->GetSurfaceHeight());
				FMatrix ProjectionMatrix;
				BuildProjectionMatrix(CaptureSize, ECameraProjectionMode::Perspective, FOV, 1.0f, ProjectionMatrix);
				FPostProcessSettings PostProcessSettings;

				float StereoIPD = 0.0f;
				if (bIsODS)
				{
					StereoIPD = (CaptureIter == 1) ? CaptureComponent->IPD * -0.5f : CaptureComponent->IPD * 0.5f;
				}

			FSceneRenderer* SceneRenderer = CreateSceneRendererForSceneCapture(this, CaptureComponent, TextureTarget->GameThread_GetRenderTargetResource(), CaptureSize, ViewRotationMatrix, Location, ProjectionMatrix, CaptureComponent->MaxViewDistanceOverride, true, &PostProcessSettings, 0, CaptureComponent->GetViewOwner(), StereoIPD);
			SceneRenderer->ViewFamily.SceneCaptureSource = SCS_SceneColorHDR;

				// [ASTRO] CaptureEpochSnapshot for cube face
				{
					TArray<FAstroCellState> FaceCells;
					FAstroCellState FaceCell;
					FaceCell.CellID         = FString::Printf(TEXT("cube_face_%d_iter_%u"), faceidx, CaptureIter);
					FaceCell.bActive        = true;
					FaceCell.LastEpochIndex = GAstroSnapshotManager.GetCurrentEpoch();
					FaceCell.SubscribedTopics = { TEXT("scene.capture.cube"), FString::Printf(TEXT("face.%d"), faceidx) };
					FaceCell.LastPublishedPayload = FString::Printf(
						TEXT("loc=(%.1f,%.1f,%.1f) face=%d ipd=%.3f"),
						Location.X, Location.Y, Location.Z, faceidx, StereoIPD);
					FaceCell.WeightVector = { static_cast<float>(faceidx), StereoIPD };
					FaceCells.Add(FaceCell);

					const int64 NowMs = (int64)(FPlatformTime::Seconds() * 1000.0);
					GAstroSnapshotManager.CaptureEpochSnapshot(FaceCells, NowMs);
				}

				FTextureRenderTargetCubeResource* TextureRenderTarget = static_cast<FTextureRenderTargetCubeResource*>(TextureTarget->GameThread_GetRenderTargetResource());
				FString EventName;
				if (!CaptureComponent->ProfilingEventName.IsEmpty())
				{
					EventName = CaptureComponent->ProfilingEventName;
				}
				else if (CaptureComponent->GetOwner())
				{
					CaptureComponent->GetOwner()->GetFName().ToString(EventName);
				}
				ENQUEUE_RENDER_COMMAND(CaptureCommand)(
					[SceneRenderer, TextureRenderTarget, EventName, TargetFace](FRHICommandListImmediate& RHICmdList)
				{
					UpdateSceneCaptureContent_RenderThread(RHICmdList, SceneRenderer, TextureRenderTarget, TextureRenderTarget, EventName, FResolveParams(FResolveRect(), TargetFace));
				}
				);
			}
		}
	}

	if (bIsODS)
	{
		const FTextureRenderTargetCubeResource* const LeftEye = static_cast<FTextureRenderTargetCubeResource*>(CaptureComponent->TextureTargetLeft->GameThread_GetRenderTargetResource());
		const FTextureRenderTargetCubeResource* const RightEye = static_cast<FTextureRenderTargetCubeResource*>(CaptureComponent->TextureTargetRight->GameThread_GetRenderTargetResource());
		FTextureRenderTargetResource* const RenderTarget = CaptureComponent->TextureTargetODS->GameThread_GetRenderTargetResource();
		const ERHIFeatureLevel::Type InFeatureLevel = FeatureLevel;

		ENQUEUE_RENDER_COMMAND(ODSCaptureCommand)(
			[LeftEye, RightEye, RenderTarget, InFeatureLevel](FRHICommandListImmediate& RHICmdList)
		{
			ODSCapture_RenderThread(RHICmdList, LeftEye, RightEye, RenderTarget, InFeatureLevel);
		}
		);
	}
}
