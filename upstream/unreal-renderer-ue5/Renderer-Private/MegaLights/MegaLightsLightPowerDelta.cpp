// Copyright Epic Games, Inc. All Rights Reserved.

#include "SceneViewState.h"
#include "SceneRendering.h"
#include "BasePassRendering.h"

static TAutoConsoleVariable<bool> CVarMegaLightsLightPowerDeltaEnable(
	TEXT("r.MegaLights.LightPowerDelta.Enable"),
	true,
	TEXT("Whether to use light power delta to drive some heuristics.\n")
	TEXT("Sampling uses it to detect light on so lights can be treated as visible immediately after they are on."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<bool> CVarMegaLightsLightPowerDeltaAsyncCompute(
	TEXT("r.MegaLights.LightPowerDelta.AsyncCompute"),
	false,
	TEXT("Whether to compute light power delta using async compute."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<bool> CVarMegaLightsLightPowerDeltaLightFunctions(
	TEXT("r.MegaLights.LightPowerDelta.LightFunctions"),
	false,
	TEXT("Whether to include light function when computing light power delta.\n")
	TEXT("It only does a single evaluation in the UV center so it may not work well for LFs animated non-uniformly."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

DECLARE_GPU_STAT(MegaLightsLightPowerDelta);

namespace MegaLights
{
	bool UseLightPowerDelta(const FViewInfo& View)
	{
		if (!View.ForwardLightingResources.ForwardLightUniformParameters
			|| !View.ForwardLightingResources.ForwardLightUniformBuffer
			|| !MegaLights::IsEnabled(*View.Family))
		{
			return false;
		}

		return CVarMegaLightsLightPowerDeltaEnable.GetValueOnRenderThread();
	}

	EPixelFormat GetLightPowerDataFormat()
	{
		return PF_R32_FLOAT;
	}
}

class FComputeLightPowerDeltaCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FComputeLightPowerDeltaCS)
	SHADER_USE_PARAMETER_STRUCT(FComputeLightPowerDeltaCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int, bUseLightFunctionAtlas)
		SHADER_PARAMETER(int, bHasPrevLightPower)
		SHADER_PARAMETER(float, HistoryPreExposureCorrection)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, RWLightPowerHistoryRatio)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, RWLightPower)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, PrevLightPowerBuffer)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FForwardLightUniformParameters, ForwardLightStruct)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(LightFunctionAtlas::FLightFunctionAtlasGlobalParameters, LightFunctionAtlas)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		FForwardLightingParameters::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}

	static int32 GetGroupSize()
	{
		return 64;
	}
};

IMPLEMENT_GLOBAL_SHADER(FComputeLightPowerDeltaCS, "/Engine/Private/MegaLights/MegaLightsLightPowerDelta.usf", "ComputeLightPowerDeltaCS", SF_Compute);

void FSceneRenderer::ComputeMegaLightsLightPowerDelta(FRDGBuilder& GraphBuilder)
{
	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLightsLightPowerDelta, "MegaLightsLightPowerDelta");

	const ERDGPassFlags ComputePassFlags = CVarMegaLightsLightPowerDeltaAsyncCompute.GetValueOnRenderThread() ? ERDGPassFlags::AsyncCompute : ERDGPassFlags::Compute;
	const EPixelFormat ElementFormat = MegaLights::GetLightPowerDataFormat();
	const uint32 ElementSize = GPixelFormats[ElementFormat].BlockBytes;

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];

		uint32 NumForwardLights = 1;
		if (View.ForwardLightingResources.ForwardLightUniformParameters)
		{
			const FForwardLightUniformParameters& ForwardLightParameters = *View.ForwardLightingResources.ForwardLightUniformParameters;
			NumForwardLights = FMath::Max(ForwardLightParameters.NumLocalLights + ForwardLightParameters.NumDirectionalLights, 1u);
		}

		View.ForwardLightingResources.LightPowerHistoryRatioBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(ElementSize, NumForwardLights), TEXT("MegaLights.LightPowerHistoryRatio"));
		FRDGBufferUAVRef LightPowerHistoryRatioUAV = GraphBuilder.CreateUAV(View.ForwardLightingResources.LightPowerHistoryRatioBuffer, ElementFormat);

		if (MegaLights::UseLightPowerDelta(View))
		{
			RDG_GPU_MASK_SCOPE(GraphBuilder, View.GPUMask);
			RDG_EVENT_SCOPE_CONDITIONAL(GraphBuilder, Views.Num() > 1, "View%d", ViewIndex);
			
			const bool bHasPrevLightPower = !View.bCameraCut && View.ViewState && View.ViewState->MegaLights.PrevLightPower;

			FRDGBufferRef PrevLightPower;
			if (bHasPrevLightPower)
			{
				PrevLightPower = GraphBuilder.RegisterExternalBuffer(View.ViewState->MegaLights.PrevLightPower);
			}
			else
			{
				PrevLightPower = GSystemTextures.GetDefaultBuffer(GraphBuilder, ElementSize, 0.0f);
			}

			FRDGBufferRef LightPower = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(ElementSize, NumForwardLights), TEXT("MegaLights.LightPower"));

			FComputeLightPowerDeltaCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FComputeLightPowerDeltaCS::FParameters>();
			PassParameters->bUseLightFunctionAtlas = CVarMegaLightsLightPowerDeltaLightFunctions.GetValueOnRenderThread() ? 1 : 0;
			PassParameters->bHasPrevLightPower = bHasPrevLightPower ? 1 : 0;
			PassParameters->HistoryPreExposureCorrection = View.PreExposure / View.PrevViewInfo.SceneColorPreExposure;
			PassParameters->RWLightPowerHistoryRatio = LightPowerHistoryRatioUAV;
			PassParameters->RWLightPower = GraphBuilder.CreateUAV(LightPower, ElementFormat);
			PassParameters->PrevLightPowerBuffer = GraphBuilder.CreateSRV(PrevLightPower, ElementFormat);
			PassParameters->View = View.ViewUniformBuffer;
			PassParameters->ForwardLightStruct = View.ForwardLightingResources.ForwardLightUniformBuffer;
			PassParameters->LightFunctionAtlas = LightFunctionAtlas::BindGlobalParameters(GraphBuilder, View);

			auto ComputeShader = View.ShaderMap->GetShader<FComputeLightPowerDeltaCS>();

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("ComputeLightPowerDelta"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FComputeShaderUtils::GetGroupCount(NumForwardLights, FComputeLightPowerDeltaCS::GetGroupSize()));

			if (View.ViewState && !View.bStatePrevViewInfoIsReadOnly)
			{
				GraphBuilder.QueueBufferExtraction(LightPower, &View.ViewState->MegaLights.PrevLightPower);
			}
		}
		else
		{
			AddClearUAVFloatPass(GraphBuilder, LightPowerHistoryRatioUAV, 1.0f);

			if (View.ViewState && !View.bStatePrevViewInfoIsReadOnly)
			{
				View.ViewState->MegaLights.PrevLightPower = nullptr;
			}
		}
	}
}
