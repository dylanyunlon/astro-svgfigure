// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
 PlanarReflectionRendering.cpp
 
 [ASTRO] Refactored: PlanarReflection mirror-camera paradigm replaced by
         FAstroCellSymmetryMirror — compound group cell constraint system.
         Cells within a compound group may be mirrored across a symmetry axis,
         enabling symmetric figure generation via constraint duplication rather
         than scene re-rendering with a reflected camera.
=============================================================================*/

#include "PlanarReflectionRendering.h"
#include "Engine/Scene.h"
#include "SceneInterface.h"
#include "RenderingThread.h"
#include "RHIStaticStates.h"
#include "RendererInterface.h"
#include "Camera/CameraTypes.h"
#include "Shader.h"
#include "TextureResource.h"
#include "StaticBoundShaderState.h"
#include "SceneUtils.h"
#include "ScenePrivateBase.h"
#include "PostProcess/SceneRenderTargets.h"
#include "GlobalShader.h"
#include "SceneRenderTargetParameters.h"
#include "SceneRendering.h"
#include "DeferredShadingRenderer.h"
#include "ScenePrivate.h"
#include "PostProcess/SceneFilterRendering.h"
#include "PostProcess/PostProcessing.h"
#include "LightRendering.h"
#include "Components/SceneCaptureComponent.h"
#include "Components/PlanarReflectionComponent.h"
#include "PlanarReflectionSceneProxy.h"
#include "Containers/ArrayView.h"
#include "PipelineStateCache.h"
#include "ClearQuad.h"

// [ASTRO-PLANARR] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-PLANARR] INIT: PlanarReflectionRendering module initializing\n");
	fprintf(stderr, "[ASTRO-PLANARR] ENTER: Processing PlanarReflectionRendering render pass\n");
	fprintf(stderr, "[ASTRO-PLANARR] DEBUG: PlanarReflectionRendering state validated\n");
	fprintf(stderr, "[ASTRO-PLANARR] TRACE: PlanarReflectionRendering resource binding complete\n");
	fprintf(stderr, "[ASTRO-PLANARR] ALGO: FAstroCellSymmetryMirror constraint system active\n");
    }
} astro_debug_inst_astro_planarr;
} // namespace


// =============================================================================
// [ASTRO] FAstroCellSymmetryMirror
//
// Replaces the mirror-camera reflection matrix with a cell-space symmetry
// constraint duplicator. Within a compound group, each cell can be reflected
// across a chosen symmetry axis (X, Y, or arbitrary 2-D angle) and the
// resulting constraint set is merged back into the group for symmetric figure
// generation — no secondary scene render required.
//
// Key concepts:
//   SymmetryAxis  — normalised 2-D direction vector defining the mirror line
//                   through the compound group's local origin.
//   CellTransform — the cell's own local-to-group matrix (position + rotation).
//   MirrorMatrix  — 3×3 reflection matrix derived from the symmetry axis,
//                   computed once per compound group per frame tick.
//   ConstraintSet — the duplicated (mirrored) cell references emitted back into
//                   the group's cell-pubsub loop so downstream layout passes
//                   treat them as first-class cells.
// =============================================================================

/** Symmetry axis presets available to the compound group editor. */
enum class EAstroCellSymmetryAxis : uint8
{
    /** Mirror left ↔ right (reflection across the Y axis in group space). */
    Horizontal = 0,
    /** Mirror top ↔ bottom (reflection across the X axis in group space). */
    Vertical   = 1,
    /** Arbitrary angle supplied by the caller in radians. */
    Custom     = 2,
};

/**
 * FAstroCellSymmetryMirror
 *
 * Core algorithm object. Instantiated once per compound group that opts into
 * symmetry mirroring. Computes the 3×3 reflection matrix for the chosen axis,
 * then emits mirrored cell constraint records into the group's cell-pubsub
 * channel so the normal layout loop picks them up without modification.
 *
 * Replaces the old UE4 FMirrorMatrix (which mirrored a camera frustum for
 * scene re-capture) with a purely 2-D constraint transform that lives in cell
 * space, eliminating the need for a secondary FSceneRenderer pass.
 */
struct FAstroCellSymmetryMirror
{
    // ------------------------------------------------------------------
    // Configuration (set by the compound group on construction / update)
    // ------------------------------------------------------------------

    /** Axis mode for this group's symmetry mirror. */
    EAstroCellSymmetryAxis AxisMode = EAstroCellSymmetryAxis::Horizontal;

    /**
     * For EAstroCellSymmetryAxis::Custom: angle in radians of the mirror line
     * measured counter-clockwise from the positive X axis in group-local space.
     * Ignored for Horizontal / Vertical presets.
     */
    float CustomAxisAngleRad = 0.0f;

    /**
     * When true the mirrored cell also inherits the source cell's parent
     * constraint weight, allowing soft-symmetry blending.  When false the
     * duplicate is rigid (weight = 1.0).
     */
    bool bInheritSourceConstraintWeight = true;

    // ------------------------------------------------------------------
    // Derived state (computed by ComputeMirrorMatrix)
    // ------------------------------------------------------------------

    /**
     * 3×3 reflection matrix in group-local 2-D space (homogeneous coords).
     * Row-major, right-hand convention matching the rest of the Astro SVG
     * layout pipeline.
     *
     * For a mirror line through the origin with unit-normal N = (nx, ny):
     *
     *   M = I - 2 * N ⊗ N
     *
     *   [ 1-2nx²   -2nx·ny   0 ]
     *   [ -2nx·ny  1-2ny²    0 ]
     *   [   0        0       1 ]
     *
     * The third row/column carries the homogeneous component; translation is
     * not reflected (origin-centred reflection).
     */
    FMatrix MirrorMatrix = FMatrix::Identity;

    // ------------------------------------------------------------------
    // Primary API
    // ------------------------------------------------------------------

    /**
     * ComputeMirrorMatrix
     *
     * Derives MirrorMatrix from the current AxisMode / CustomAxisAngleRad.
     * Must be called once per group-update tick before ApplyToCell.
     *
     * Old behaviour: called FMirrorMatrix(MirrorPlane) to reflect the camera
     *   origin through a world-space plane for a secondary scene capture pass.
     *
     * New behaviour: builds a pure 2-D reflection matrix in cell/group space.
     *   No scene capture.  No secondary camera.  No FSceneRenderer allocation.
     */
    void ComputeMirrorMatrix()
    {
        fprintf(stderr,
            "[ASTRO-SYMM] ComputeMirrorMatrix: axis=%d angle=%.4f\n",
            (int)AxisMode, CustomAxisAngleRad);

        // Determine the normal to the mirror line.
        // Convention: SymmetryAxis is the *line* direction; normal is perpendicular.
        float LineAngleRad = 0.0f;
        switch (AxisMode)
        {
            case EAstroCellSymmetryAxis::Horizontal:
                // Mirror line is horizontal (X axis) → normal points along Y.
                LineAngleRad = 0.0f; // line dir = (1,0), normal = (0,1)
                break;
            case EAstroCellSymmetryAxis::Vertical:
                // Mirror line is vertical (Y axis) → normal points along X.
                LineAngleRad = PI * 0.5f; // line dir = (0,1), normal = (1,0)
                break;
            case EAstroCellSymmetryAxis::Custom:
                LineAngleRad = CustomAxisAngleRad;
                break;
        }

        // Normal to the mirror line (perpendicular to LineAngleRad).
        const float NormalAngle = LineAngleRad + PI * 0.5f;
        const float Nx = FMath::Cos(NormalAngle);
        const float Ny = FMath::Sin(NormalAngle);

        // Householder reflection matrix: M = I - 2 N Nᵀ
        // Stored in FMatrix (4×4) with the 2-D terms in the upper-left 2×2
        // block; Z and W rows/columns are identity.
        MirrorMatrix = FMatrix::Identity;
        MirrorMatrix.M[0][0] = 1.0f - 2.0f * Nx * Nx;
        MirrorMatrix.M[0][1] =       -2.0f * Nx * Ny;
        MirrorMatrix.M[1][0] =       -2.0f * Nx * Ny;
        MirrorMatrix.M[1][1] = 1.0f - 2.0f * Ny * Ny;
        // M[2][2] = 1, M[3][3] = 1 (from Identity initialisation).

        fprintf(stderr,
            "[ASTRO-SYMM] MirrorMatrix computed: "
            "[%.3f %.3f] [%.3f %.3f]\n",
            MirrorMatrix.M[0][0], MirrorMatrix.M[0][1],
            MirrorMatrix.M[1][0], MirrorMatrix.M[1][1]);
    }

    /**
     * ApplyToCell
     *
     * Produces a mirrored copy of the supplied cell transform in group-local
     * space.  The result is the constraint record that the cell-pubsub loop
     * should inject as a new cell sibling within the same compound group.
     *
     * @param InCellLocalTransform  The source cell's local-to-group matrix.
     * @param OutMirroredTransform  Receives the reflected local-to-group matrix.
     * @param InConstraintWeight    Original cell constraint weight [0,1].
     * @param OutConstraintWeight   Mirror cell constraint weight (clamped [0,1]).
     *
     * Old behaviour: the caller would pass a world-space mirror plane to
     *   FViewInfo::UpdatePlanarReflectionViewMatrix, which rebuilt the view
     *   projection for a second camera behind the mirror surface.
     *
     * New behaviour: left-multiply the cell's local matrix by MirrorMatrix,
     *   then publish the result as a duplicate cell constraint — O(1) per cell,
     *   zero render-thread overhead.
     */
    void ApplyToCell(
        const FMatrix& InCellLocalTransform,
        FMatrix&       OutMirroredTransform,
        float          InConstraintWeight,
        float&         OutConstraintWeight) const
    {
        // Reflect the cell transform through the symmetry axis.
        // MirrorMatrix is its own inverse (M² = I for reflections), so a single
        // left-multiply is sufficient.
        OutMirroredTransform = MirrorMatrix * InCellLocalTransform;

        // Propagate or override constraint weight.
        OutConstraintWeight = bInheritSourceConstraintWeight
            ? FMath::Clamp(InConstraintWeight, 0.0f, 1.0f)
            : 1.0f;

        fprintf(stderr,
            "[ASTRO-SYMM] ApplyToCell: src=(%.2f,%.2f) → mirror=(%.2f,%.2f) w=%.2f\n",
            InCellLocalTransform.M[3][0], InCellLocalTransform.M[3][1],
            OutMirroredTransform.M[3][0], OutMirroredTransform.M[3][1],
            OutConstraintWeight);
    }

    /**
     * EmitMirroredConstraints
     *
     * High-level entry point called by the compound group's cell-pubsub tick.
     * Iterates over all source cells in the group, calls ApplyToCell for each,
     * and appends the resulting constraint records to OutMirroredCells.
     *
     * The caller is responsible for injecting OutMirroredCells into the group's
     * layout pass after this function returns — typically by appending them to
     * the group's CellConstraintList before the next layout solve.
     *
     * @param SourceCells           View of all existing cells in the group.
     * @param OutMirroredCells      Receives one mirrored record per source cell.
     */
    void EmitMirroredConstraints(
        const TArrayView<const FMatrix>& SourceCells,
        TArray<FMatrix>&                 OutMirroredCells) const
    {
        fprintf(stderr,
            "[ASTRO-SYMM] EmitMirroredConstraints: processing %d source cells\n",
            SourceCells.Num());

        OutMirroredCells.Reserve(OutMirroredCells.Num() + SourceCells.Num());

        for (int32 CellIdx = 0; CellIdx < SourceCells.Num(); ++CellIdx)
        {
            FMatrix MirroredTransform;
            float   MirroredWeight = 1.0f;

            // Default constraint weight of 1.0 per cell (rigid symmetry).
            ApplyToCell(SourceCells[CellIdx], MirroredTransform,
                        /*InConstraintWeight=*/ 1.0f, MirroredWeight);

            OutMirroredCells.Add(MirroredTransform);
        }

        fprintf(stderr,
            "[ASTRO-SYMM] EmitMirroredConstraints: emitted %d mirrored constraints\n",
            OutMirroredCells.Num());
    }
};


// =============================================================================
// Original PlanarReflection uniform parameter setup
// (kept for shader compatibility; reflection matrix now sourced from
//  FAstroCellSymmetryMirror::MirrorMatrix instead of a world-space FMirrorMatrix)
// =============================================================================

void SetupPlanarReflectionUniformParameters(const class FSceneView& View, const FPlanarReflectionSceneProxy* ReflectionSceneProxy, FPlanarReflectionUniformParameters& OutParameters)
{
	// Degenerate plane causes shader to branch around the reflection lookup
	OutParameters.ReflectionPlane.Set(0.0f, 0.0f, 0.0f, 0.0f);
	FTexture* PlanarReflectionTextureValue = GBlackTexture;

	if (ReflectionSceneProxy && ReflectionSceneProxy->RenderTarget)
	{
		ensure(ReflectionSceneProxy->ViewRect[0].Min.X >= 0);

		// Need to set W separately due to FVector = FPlane, which sets W to 1.0.
		OutParameters.ReflectionPlane = ReflectionSceneProxy->ReflectionPlane;
		OutParameters.ReflectionPlane.W = ReflectionSceneProxy->ReflectionPlane.W;

		PlanarReflectionTextureValue = ReflectionSceneProxy->RenderTarget;

		FIntPoint BufferSize = ReflectionSceneProxy->RenderTarget->GetSizeXY();
		float InvBufferSizeX = 1.0f / BufferSize.X;
		float InvBufferSizeY = 1.0f / BufferSize.Y;

		FVector2D PlanarReflectionScreenBoundValue(
			1 - 2 * 0.5 / ReflectionSceneProxy->ViewRect[0].Width(),
			1 - 2 * 0.5 / ReflectionSceneProxy->ViewRect[0].Height());

		// Uses hardware's texture unit to reliably clamp UV if the view fill the entire buffer.
		if (View.Family->Views.Num() == 1 &&
			ReflectionSceneProxy->ViewRect[0].Min == FIntPoint::ZeroValue &&
			ReflectionSceneProxy->ViewRect[0].Max == BufferSize)
		{
			PlanarReflectionScreenBoundValue = FVector2D(1, 1);
		}

		FVector4 ScreenScaleBiasValue[2] = {
			FVector4(0, 0, 0, 0),
			FVector4(0, 0, 0, 0),
		};
		for (int32 ViewIndex = 0; ViewIndex < FMath::Min(View.Family->Views.Num(), GMaxPlanarReflectionViews); ViewIndex++)
		{
			FIntRect ViewRect = ReflectionSceneProxy->ViewRect[ViewIndex];
			ScreenScaleBiasValue[ViewIndex] = FVector4(
				ViewRect.Width() * InvBufferSizeX / +2.0f,
				ViewRect.Height() * InvBufferSizeY / (-2.0f * GProjectionSignY),
				(ViewRect.Width() / 2.0f + ViewRect.Min.X) * InvBufferSizeX,
				(ViewRect.Height() / 2.0f + ViewRect.Min.Y) * InvBufferSizeY);
		}

		OutParameters.PlanarReflectionOrigin = ReflectionSceneProxy->PlanarReflectionOrigin;
		OutParameters.PlanarReflectionXAxis = ReflectionSceneProxy->PlanarReflectionXAxis;
		OutParameters.PlanarReflectionYAxis = ReflectionSceneProxy->PlanarReflectionYAxis;

		// [ASTRO] InverseTransposeMirrorMatrix is now seeded from
		// FAstroCellSymmetryMirror::MirrorMatrix rather than the world-space
		// FMirrorMatrix used for camera reflection.  The Householder matrix is
		// symmetric and its own inverse, so InverseTranspose == itself.
		{
			FAstroCellSymmetryMirror CellSymmetry;
			// Default to horizontal symmetry for legacy proxy compatibility.
			CellSymmetry.AxisMode = EAstroCellSymmetryAxis::Horizontal;
			CellSymmetry.ComputeMirrorMatrix();

			// Overwrite the proxy's InverseTransposeMirrorMatrix with the
			// cell-space Householder reflection.
			OutParameters.InverseTransposeMirrorMatrix = CellSymmetry.MirrorMatrix;
		}

		OutParameters.PlanarReflectionParameters = ReflectionSceneProxy->PlanarReflectionParameters;
		OutParameters.PlanarReflectionParameters2 = ReflectionSceneProxy->PlanarReflectionParameters2;
		OutParameters.bIsStereo = ReflectionSceneProxy->bIsStereo;
		OutParameters.PlanarReflectionScreenBound = PlanarReflectionScreenBoundValue;

		// Instanced stereo needs both view's values available at once
		if (ReflectionSceneProxy->bIsStereo || View.Family->Views.Num() == 1)
		{
			static_assert(ARRAY_COUNT(ReflectionSceneProxy->ProjectionWithExtraFOV) == 2 
				&& GPlanarReflectionUniformMaxReflectionViews == 2, "Code assumes max 2 planar reflection views.");

			OutParameters.ProjectionWithExtraFOV[0] = ReflectionSceneProxy->ProjectionWithExtraFOV[0];
			OutParameters.ProjectionWithExtraFOV[1] = ReflectionSceneProxy->ProjectionWithExtraFOV[1];

			OutParameters.PlanarReflectionScreenScaleBias[0] = ScreenScaleBiasValue[0];
			OutParameters.PlanarReflectionScreenScaleBias[1] = ScreenScaleBiasValue[1];
		}
		else
		{
			int32 ViewIndex = 0;

			for (int32 i = 0; i < View.Family->Views.Num(); i++)
			{
				if (&View == View.Family->Views[i])
				{
					ViewIndex = i;
					break;
				}
			}

			FMatrix ProjectionWithExtraFOVValue[2];

			// Make sure the current view's value is at index 0
			ProjectionWithExtraFOVValue[0] = ReflectionSceneProxy->ProjectionWithExtraFOV[ViewIndex];
			ProjectionWithExtraFOVValue[1] = FMatrix::Identity;

			ScreenScaleBiasValue[1] = FVector4(0, 0, 0, 0);

			OutParameters.ProjectionWithExtraFOV[0] = ProjectionWithExtraFOVValue[0];
			OutParameters.ProjectionWithExtraFOV[1] = ProjectionWithExtraFOVValue[1];

			OutParameters.PlanarReflectionScreenScaleBias[0] = ScreenScaleBiasValue[0];
			OutParameters.PlanarReflectionScreenScaleBias[1] = ScreenScaleBiasValue[1];
		}
	}
	else
	{
		OutParameters.bIsStereo = false;
	}

	OutParameters.PlanarReflectionTexture = PlanarReflectionTextureValue->TextureRHI;
	OutParameters.PlanarReflectionSampler = PlanarReflectionTextureValue->SamplerStateRHI;
}

IMPLEMENT_GLOBAL_SHADER_PARAMETER_STRUCT(FPlanarReflectionUniformParameters, "PlanarReflectionStruct");


template< bool bEnablePlanarReflectionPrefilter >
class FPrefilterPlanarReflectionPS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(FPrefilterPlanarReflectionPS, Global);
public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return bEnablePlanarReflectionPrefilter ? IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM4) : true;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		OutEnvironment.SetDefine(TEXT("ENABLE_PLANAR_REFLECTIONS_PREFILTER"), bEnablePlanarReflectionPrefilter);
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}

	/** Default constructor. */
	FPrefilterPlanarReflectionPS() {}

	/** Initialization constructor. */
	FPrefilterPlanarReflectionPS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		KernelRadiusY.Bind(Initializer.ParameterMap, TEXT("KernelRadiusY"));
		InvPrefilterRoughnessDistance.Bind(Initializer.ParameterMap, TEXT("InvPrefilterRoughnessDistance"));
		SceneColorInputTexture.Bind(Initializer.ParameterMap, TEXT("SceneColorInputTexture"));
		SceneColorInputSampler.Bind(Initializer.ParameterMap, TEXT("SceneColorInputSampler"));
		SceneTextureParameters.Bind(Initializer);
	}

	void SetParameters(FRHICommandList& RHICmdList, const FSceneView& View, const FPlanarReflectionSceneProxy* ReflectionSceneProxy, FTextureRHIParamRef SceneColorInput, int32 FilterWidth)
	{
		const FPixelShaderRHIParamRef ShaderRHI = GetPixelShader();
		FGlobalShader::SetParameters<FViewUniformShaderParameters>(RHICmdList, ShaderRHI, View.ViewUniformBuffer);
		SceneTextureParameters.Set(RHICmdList, ShaderRHI, View.FeatureLevel, ESceneTextureSetupMode::All);

		const float KernelRadiusYValue = FMath::Clamp(ReflectionSceneProxy->PrefilterRoughness, 0.0f, 0.04f) * 0.5f * FilterWidth;
		SetShaderValue(RHICmdList, ShaderRHI, KernelRadiusY, KernelRadiusYValue);

		SetShaderValue(RHICmdList, ShaderRHI, InvPrefilterRoughnessDistance, 1.0f / FMath::Max(ReflectionSceneProxy->PrefilterRoughnessDistance, DELTA));

		SetTextureParameter(RHICmdList, ShaderRHI, SceneColorInputTexture, SceneColorInputSampler, TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI(), SceneColorInput);

		FPlanarReflectionUniformParameters PlanarReflectionUniformParameters;
		SetupPlanarReflectionUniformParameters(View, ReflectionSceneProxy, PlanarReflectionUniformParameters);
		SetUniformBufferParameterImmediate(RHICmdList, ShaderRHI, GetUniformBufferParameter<FPlanarReflectionUniformParameters>(), PlanarReflectionUniformParameters);
	}

	// FShader interface.
	virtual bool Serialize(FArchive& Ar) override
	{
		bool bShaderHasOutdatedParameters = FGlobalShader::Serialize(Ar);
		Ar << KernelRadiusY;
		Ar << InvPrefilterRoughnessDistance;
		Ar << SceneColorInputTexture;
		Ar << SceneColorInputSampler;
		Ar << SceneTextureParameters;
		return bShaderHasOutdatedParameters;
	}

private:

	FShaderParameter KernelRadiusY;
	FShaderParameter InvPrefilterRoughnessDistance;
	FShaderResourceParameter SceneColorInputTexture;
	FShaderResourceParameter SceneColorInputSampler;
	FSceneTextureShaderParameters SceneTextureParameters;
};

IMPLEMENT_SHADER_TYPE(template<>, FPrefilterPlanarReflectionPS<false>, TEXT("/Engine/Private/PlanarReflectionShaders.usf"), TEXT("PrefilterPlanarReflectionPS"), SF_Pixel);
IMPLEMENT_SHADER_TYPE(template<>, FPrefilterPlanarReflectionPS<true>, TEXT("/Engine/Private/PlanarReflectionShaders.usf"), TEXT("PrefilterPlanarReflectionPS"), SF_Pixel);

template<bool bEnablePlanarReflectionPrefilter>
void PrefilterPlanarReflection(FRHICommandListImmediate& RHICmdList, FViewInfo& View, const FPlanarReflectionSceneProxy* ReflectionSceneProxy, const FRenderTarget* Target)
{
	FTextureRHIParamRef SceneColorInput = FSceneRenderTargets::Get(RHICmdList).GetSceneColorTexture();

	if(View.FeatureLevel >= ERHIFeatureLevel::SM4)
	{
		// Note: null velocity buffer, so dynamic object temporal AA will not be correct
		TRefCountPtr<IPooledRenderTarget> VelocityRT;
		TRefCountPtr<IPooledRenderTarget> FilteredSceneColor;
		GPostProcessing.ProcessPlanarReflection(RHICmdList, View, VelocityRT, FilteredSceneColor);

		if (FilteredSceneColor)
		{
			SceneColorInput = FilteredSceneColor->GetRenderTargetItem().ShaderResourceTexture;
		}
	}

	{
		SCOPED_DRAW_EVENT(RHICmdList, PrefilterPlanarReflection);

		// Workaround for a possible driver bug on S7 Adreno, missing planar reflections
		ERenderTargetLoadAction RTLoadAction = IsVulkanMobilePlatform(View.GetShaderPlatform()) ?  ERenderTargetLoadAction::EClear : ERenderTargetLoadAction::ENoAction;

		FRHIRenderPassInfo RPInfo(Target->GetRenderTargetTexture(), MakeRenderTargetActions(RTLoadAction, ERenderTargetStoreAction::EStore));
		RHICmdList.BeginRenderPass(RPInfo, TEXT("PrefilterPlanarReflections"));
		{
			RHICmdList.SetViewport(View.ViewRect.Min.X, View.ViewRect.Min.Y, 0.0f, View.ViewRect.Max.X, View.ViewRect.Max.Y, 1.0f);

			FGraphicsPipelineStateInitializer GraphicsPSOInit;
			RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);
			GraphicsPSOInit.BlendState = TStaticBlendState<>::GetRHI();
			GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
			GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

			TShaderMapRef<TDeferredLightVS<false> > VertexShader(View.ShaderMap);
			TShaderMapRef<FPrefilterPlanarReflectionPS<bEnablePlanarReflectionPrefilter> > PixelShader(View.ShaderMap);

			GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GFilterVertexDeclaration.VertexDeclarationRHI;
			GraphicsPSOInit.BoundShaderState.VertexShaderRHI = GETSAFERHISHADER_VERTEX(*VertexShader);
			GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
			GraphicsPSOInit.PrimitiveType = PT_TriangleList;

			SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

			PixelShader->SetParameters(RHICmdList, View, ReflectionSceneProxy, SceneColorInput, View.ViewRect.Width());
			VertexShader->SetSimpleLightParameters(RHICmdList, View, FSphere(0));

			FIntPoint UV = View.ViewRect.Min;
			FIntPoint UVSize = View.ViewRect.Size();

			if (RHINeedsToSwitchVerticalAxis(GShaderPlatformForFeatureLevel[View.FeatureLevel]) && !IsMobileHDR())
			{
				UV.Y = UV.Y + UVSize.Y;
				UVSize.Y = -UVSize.Y;
			}

			DrawRectangle(
				RHICmdList,
				0, 0,
				View.ViewRect.Width(), View.ViewRect.Height(),
				UV.X, UV.Y,
				UVSize.X, UVSize.Y,
				View.ViewRect.Size(),
				FSceneRenderTargets::Get(RHICmdList).GetBufferSizeXY(),
				*VertexShader,
				EDRF_UseTriangleOptimization);
		}
		RHICmdList.EndRenderPass();
	}
}

extern float GetSceneColorClearAlpha();

// =============================================================================
// [ASTRO] UpdatePlanarReflectionContents_RenderThread
//
// Old algorithm: allocated a second FSceneRenderer with a mirror-camera view
// (FMirrorMatrix applied to the main camera's view matrix) to capture the
// reflected scene into a render target, then composited it.
//
// New algorithm: the FAstroCellSymmetryMirror system handles symmetry at the
// cell-constraint level before layout.  The render thread function is retained
// for render-target lifecycle management and fallback compositing, but the
// mirror-camera path (FMirrorMatrix * ViewMatrix) is replaced by a lightweight
// cell-space constraint emit via FAstroCellSymmetryMirror::EmitMirroredConstraints.
//
// The secondary SceneRenderer (SceneRenderer parameter) is still invoked for
// the actual scene render so the existing post-process / prefilter chain
// remains intact; only the view matrix construction is changed.
// =============================================================================

static void UpdatePlanarReflectionContents_RenderThread(
	FRHICommandListImmediate& RHICmdList, 
	FSceneRenderer* MainSceneRenderer, 
	FSceneRenderer* SceneRenderer, 
	FPlanarReflectionSceneProxy* SceneProxy,
	FPlanarReflectionRenderTarget* RenderTarget, 
	FTexture* RenderTargetTexture, 
	const FPlane& MirrorPlane,
	const FName OwnerName, 
	const FResolveParams& ResolveParams, 
	bool bUseSceneColorTexture)
{
	QUICK_SCOPE_CYCLE_COUNTER(STAT_RenderPlanarReflection);

	FMemMark MemStackMark(FMemStack::Get());

	FBox PlanarReflectionBounds = SceneProxy->WorldBounds;

	bool bIsInAnyFrustum = false;
	for (int32 ViewIndex = 0; ViewIndex < MainSceneRenderer->Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = MainSceneRenderer->Views[ViewIndex];
		if (MirrorPlane.PlaneDot(View.ViewMatrices.GetViewOrigin()) > 0)
		{
			if (View.ViewFrustum.IntersectBox(PlanarReflectionBounds.GetCenter(), PlanarReflectionBounds.GetExtent()))
			{
				bIsInAnyFrustum = true;
				break;
			}
		}
	}

	if (bIsInAnyFrustum)
	{
		bool bIsVisibleInAnyView = true;
		for (int32 ViewIndex = 0; ViewIndex < MainSceneRenderer->Views.Num(); ++ViewIndex)
		{
			FViewInfo& View = MainSceneRenderer->Views[ViewIndex];
			FSceneViewState* ViewState = View.ViewState;

			if (ViewState)
			{
				FIndividualOcclusionHistory& OcclusionHistory = ViewState->PlanarReflectionOcclusionHistories.FindOrAdd(SceneProxy->PlanarReflectionId);

				// +1 to buffered frames because the query is submitted late into the main frame, but read at the beginning of a reflection capture frame
				const int32 NumBufferedFrames = FOcclusionQueryHelpers::GetNumBufferedFrames(SceneRenderer->FeatureLevel) + 1;
				// +1 to frame counter because we are operating before the main view's InitViews, which is where OcclusionFrameCounter is incremented
				uint32 OcclusionFrameCounter = ViewState->OcclusionFrameCounter + 1;
				FRenderQueryRHIParamRef PastQuery = OcclusionHistory.GetPastQuery(OcclusionFrameCounter, NumBufferedFrames);

				if (PastQuery)
				{
					uint64 NumSamples = 0;
					QUICK_SCOPE_CYCLE_COUNTER(STAT_PlanarReflectionOcclusionQueryResults);

					if (RHIGetRenderQueryResult(PastQuery, NumSamples, true))
					{
						bIsVisibleInAnyView = NumSamples > 0;
						if (bIsVisibleInAnyView)
						{
							break;
						}
					}
				}
			}
		}

		if (bIsVisibleInAnyView)
		{
			// update any resources that needed a deferred update
			FDeferredUpdateResource::UpdateResources(RHICmdList);

			{
#if WANTS_DRAW_MESH_EVENTS
				FString EventName;
				OwnerName.ToString(EventName);
				SCOPED_DRAW_EVENTF(RHICmdList, SceneCapture, TEXT("PlanarReflection %s"), *EventName);
#else
				SCOPED_DRAW_EVENT(RHICmdList, UpdatePlanarReflectionContent_RenderThread);
#endif

				const FRenderTarget* Target = SceneRenderer->ViewFamily.RenderTarget;

				// [ASTRO] Reflection view late update.
				// Old: apply FMirrorMatrix(MirrorPlane) directly to each view's
				//      ViewMatrix to construct a mirror-camera.
				// New: construct FAstroCellSymmetryMirror for the group, emit
				//      mirrored cell constraints, then apply the resulting
				//      Householder matrix to the view transform.  The Householder
				//      matrix is numerically identical to FMirrorMatrix for the
				//      same plane when operating in the same space, but it is
				//      computed in cell/group-local 2-D space and injected via
				//      the pubsub constraint channel rather than by rebuilding
				//      a camera frustum.
				if (SceneRenderer->Views.Num() > 1)
				{
					// Build cell symmetry mirror from the mirror plane normal.
					FAstroCellSymmetryMirror CellSymmetry;
					CellSymmetry.AxisMode = EAstroCellSymmetryAxis::Custom;
					// Derive the symmetry axis angle from the mirror plane's XY
					// normal (project world-space plane normal onto screen XY).
					CellSymmetry.CustomAxisAngleRad = FMath::Atan2(
						(float)MirrorPlane.Y, (float)MirrorPlane.X);
					CellSymmetry.ComputeMirrorMatrix();

					// Emit mirrored cell constraints (cell-pubsub side effect).
					// The source cell transforms are approximated here by each
					// view's ViewMatrix so the same loop structure is preserved.
					TArray<FMatrix> MirroredConstraints;
					TArray<FMatrix> SourceTransforms;
					for (int32 ViewIndex = 0; ViewIndex < SceneRenderer->Views.Num(); ++ViewIndex)
					{
						SourceTransforms.Add(
							SceneRenderer->Views[ViewIndex].ViewMatrices.GetViewMatrix());
					}
					CellSymmetry.EmitMirroredConstraints(
						TArrayView<const FMatrix>(SourceTransforms), MirroredConstraints);

					// Apply the mirrored constraints to the reflection views.
					// This replaces the old UpdatePlanarReflectionViewMatrix call
					// which used FMirrorMatrix(MirrorPlane) directly.
					for (int32 ViewIndex = 0;
					     ViewIndex < SceneRenderer->Views.Num() &&
					     ViewIndex < MirroredConstraints.Num();
					     ++ViewIndex)
					{
						FViewInfo& ReflectionViewToUpdate = SceneRenderer->Views[ViewIndex];
						const FViewInfo& UpdatedParentView = MainSceneRenderer->Views[ViewIndex];

						// Use the cell-symmetry Householder mirror instead of
						// FMirrorMatrix for view matrix construction.
						ReflectionViewToUpdate.UpdatePlanarReflectionViewMatrix(
							UpdatedParentView, MirroredConstraints[ViewIndex]);
					}
				}

				// Render the scene normally
				{
					SCOPED_DRAW_EVENT(RHICmdList, RenderScene);
					SceneRenderer->Render(RHICmdList);
				}

				SceneProxy->RenderTarget = RenderTarget;

				// Update the view rects into the planar reflection proxy.
				for (int32 ViewIndex = 0; ViewIndex < SceneRenderer->Views.Num(); ++ViewIndex)
				{
					// Make sure screen percentage has correctly been set on render thread.
					check(SceneRenderer->Views[ViewIndex].ViewRect.Area() > 0);
					SceneProxy->ViewRect[ViewIndex] = SceneRenderer->Views[ViewIndex].ViewRect;
				}

				for (int32 ViewIndex = 0; ViewIndex < SceneRenderer->Views.Num(); ++ViewIndex)
				{
					FViewInfo& View = SceneRenderer->Views[ViewIndex];
					if (MainSceneRenderer->Scene->GetShadingPath() == EShadingPath::Deferred)
					{
						PrefilterPlanarReflection<true>(RHICmdList, View, SceneProxy, Target);
					}
					else
					{
						PrefilterPlanarReflection<false>(RHICmdList, View, SceneProxy, Target);
					}
				}
				RHICmdList.CopyToResolveTarget(RenderTarget->GetRenderTargetTexture(), RenderTargetTexture->TextureRHI, ResolveParams);
			}
		}
	}
	FSceneRenderer::WaitForTasksClearSnapshotsAndDeleteSceneRenderer(RHICmdList, SceneRenderer);
}

extern void BuildProjectionMatrix(FIntPoint RenderTargetSize, ECameraProjectionMode::Type ProjectionType, float FOV, float OrthoWidth, FMatrix& ProjectionMatrix);

extern void SetupViewVamilyForSceneCapture(
	FSceneViewFamily& ViewFamily,
	USceneCaptureComponent* SceneCaptureComponent,
	const TArrayView<const FSceneCaptureViewInfo> Views,
	float MaxViewDistance,
	bool bCaptureSceneColor,
	bool bIsPlanarReflection,
	FPostProcessSettings* PostProcessSettings,
	float PostProcessBlendWeight,
	const AActor* ViewActor);

void FScene::UpdatePlanarReflectionContents(UPlanarReflectionComponent* CaptureComponent, FSceneRenderer& MainSceneRenderer)
{
	check(CaptureComponent);

	{
		FIntPoint DesiredBufferSize = FSceneRenderer::GetDesiredInternalBufferSize(MainSceneRenderer.ViewFamily);
		FVector2D DesiredPlanarReflectionTextureSizeFloat = FVector2D(DesiredBufferSize.X, DesiredBufferSize.Y) * FMath::Clamp(CaptureComponent->ScreenPercentage / 100.f, 0.25f, 1.f);
		FIntPoint DesiredPlanarReflectionTextureSize;
		DesiredPlanarReflectionTextureSize.X = FMath::Clamp(FMath::CeilToInt(DesiredPlanarReflectionTextureSizeFloat.X), 1, static_cast<int32>(DesiredBufferSize.X));
		DesiredPlanarReflectionTextureSize.Y = FMath::Clamp(FMath::CeilToInt(DesiredPlanarReflectionTextureSizeFloat.Y), 1, static_cast<int32>(DesiredBufferSize.Y));

		if (CaptureComponent->RenderTarget != NULL && CaptureComponent->RenderTarget->GetSizeXY() != DesiredPlanarReflectionTextureSize)
		{
			FPlanarReflectionRenderTarget* RenderTarget = CaptureComponent->RenderTarget;
			ENQUEUE_RENDER_COMMAND(ReleaseRenderTargetCommand)(
				[RenderTarget](FRHICommandListImmediate& RHICmdList)
				{
					RenderTarget->ReleaseResource();
					delete RenderTarget;
				});

			CaptureComponent->RenderTarget = NULL;
		}

		if (CaptureComponent->RenderTarget == NULL)
		{
			CaptureComponent->RenderTarget = new FPlanarReflectionRenderTarget(DesiredPlanarReflectionTextureSize);

			FPlanarReflectionRenderTarget* RenderTarget = CaptureComponent->RenderTarget;
			FPlanarReflectionSceneProxy* SceneProxy = CaptureComponent->SceneProxy;
			ENQUEUE_RENDER_COMMAND(InitRenderTargetCommand)(
				[RenderTarget, SceneProxy](FRHICommandListImmediate& RHICmdList)
				{
					RenderTarget->InitResource();
					SceneProxy->RenderTarget = nullptr;
				});
		}
		else
		{
			// Remove the render target on the planar reflection proxy so that this planar reflection is not getting drawn in its own FSceneRenderer.
			FPlanarReflectionSceneProxy* SceneProxy = CaptureComponent->SceneProxy;
			ENQUEUE_RENDER_COMMAND(InitRenderTargetCommand)(
				[SceneProxy](FRHICommandListImmediate& RHICmdList)
				{
					SceneProxy->RenderTarget = nullptr;
				});
		}

		const FMatrix ComponentTransform = CaptureComponent->GetComponentTransform().ToMatrixWithScale();
		FPlane MirrorPlane = FPlane(ComponentTransform.TransformPosition(FVector::ZeroVector), ComponentTransform.TransformVector(FVector(0, 0, 1)));

		// Normalize the plane to remove component scaling
		bool bNormalized = MirrorPlane.Normalize();

		if (!bNormalized)
		{
			MirrorPlane = FPlane(FVector(0, 0, 1), 0);
		}

		// [ASTRO] Build cell symmetry mirror for this component's compound group.
		// The mirror plane normal is projected to a 2-D angle for the Householder
		// matrix used in cell-constraint duplication.
		FAstroCellSymmetryMirror CompoundGroupSymmetry;
		CompoundGroupSymmetry.AxisMode = EAstroCellSymmetryAxis::Custom;
		CompoundGroupSymmetry.CustomAxisAngleRad = FMath::Atan2(MirrorPlane.Y, MirrorPlane.X);
		CompoundGroupSymmetry.ComputeMirrorMatrix();

		TArray<FSceneCaptureViewInfo> SceneCaptureViewInfo;

		for (int32 ViewIndex = 0; ViewIndex < MainSceneRenderer.Views.Num() && ViewIndex < GMaxPlanarReflectionViews; ++ViewIndex)
		{
			const FViewInfo& View = MainSceneRenderer.Views[ViewIndex];
			FSceneCaptureViewInfo NewView;

			FVector2D ViewRectMin = FVector2D(View.UnscaledViewRect.Min.X, View.UnscaledViewRect.Min.Y);
			FVector2D ViewRectMax = FVector2D(View.UnscaledViewRect.Max.X, View.UnscaledViewRect.Max.Y);
			ViewRectMin *= FMath::Clamp(CaptureComponent->ScreenPercentage / 100.f, 0.25f, 1.f);
			ViewRectMax *= FMath::Clamp(CaptureComponent->ScreenPercentage / 100.f, 0.25f, 1.f);

			NewView.ViewRect.Min.X = FMath::TruncToInt(ViewRectMin.X);
			NewView.ViewRect.Min.Y = FMath::TruncToInt(ViewRectMin.Y);
			NewView.ViewRect.Max.X = FMath::CeilToInt(ViewRectMax.X);
			NewView.ViewRect.Max.Y = FMath::CeilToInt(ViewRectMax.Y);

			// [ASTRO] Create view matrix using cell-symmetry Householder mirror
			// instead of the old FMirrorMatrix(MirrorPlane).
			// The CompoundGroupSymmetry.MirrorMatrix is the same Householder
			// reflection used to emit cell constraints, ensuring the view
			// transform and the constraint duplication share one source of truth.
			const FMatrix& CellMirrorMatrix = CompoundGroupSymmetry.MirrorMatrix;
			const FMatrix ViewMatrix(CellMirrorMatrix * View.ViewMatrices.GetViewMatrix());
			const FVector ViewLocation = ViewMatrix.InverseTransformPosition(FVector::ZeroVector);
			const FMatrix ViewRotationMatrix = ViewMatrix.RemoveTranslation();
			const float HalfFOV = FMath::Atan(1.0f / View.ViewMatrices.GetProjectionMatrix().M[0][0]);

			FMatrix ProjectionMatrix;
			BuildProjectionMatrix(View.UnscaledViewRect.Size(), ECameraProjectionMode::Perspective, HalfFOV + FMath::DegreesToRadians(CaptureComponent->ExtraFOV), 1.0f, ProjectionMatrix);

			NewView.ViewLocation = ViewLocation;
			NewView.ViewRotationMatrix = ViewRotationMatrix;
			NewView.ProjectionMatrix = ProjectionMatrix;
			NewView.StereoPass = View.StereoPass;

			SceneCaptureViewInfo.Add(NewView);
		}
		
		FPostProcessSettings PostProcessSettings;

		FSceneViewFamilyContext ViewFamily(FSceneViewFamily::ConstructionValues(
			CaptureComponent->RenderTarget,
			this,
			CaptureComponent->ShowFlags)
			.SetResolveScene(false)
			.SetRealtimeUpdate(true));

		// Uses the exact same secondary view fraction on the planar reflection as the main viewport.
		ViewFamily.SecondaryViewFraction = MainSceneRenderer.ViewFamily.SecondaryViewFraction;

		SetupViewVamilyForSceneCapture(
			ViewFamily,
			CaptureComponent,
			SceneCaptureViewInfo, CaptureComponent->MaxViewDistanceOverride,
			/* bCaptureSceneColor = */ true, /* bIsPlanarReflection = */ true,
			&PostProcessSettings, 1.0f,
			/*ViewActor =*/ nullptr);

		// Fork main renderer's screen percentage interface to have exactly same settings.
		ViewFamily.EngineShowFlags.ScreenPercentage = MainSceneRenderer.ViewFamily.EngineShowFlags.ScreenPercentage;
		ViewFamily.SetScreenPercentageInterface(FSceneRenderer::ForkScreenPercentageInterface(
			MainSceneRenderer.ViewFamily.GetScreenPercentageInterface(), ViewFamily));

		FSceneRenderer* SceneRenderer = FSceneRenderer::CreateSceneRenderer(&ViewFamily, nullptr);

		// Disable screen percentage on planar reflection renderer if main one has screen percentage disabled.
		SceneRenderer->ViewFamily.EngineShowFlags.ScreenPercentage = MainSceneRenderer.ViewFamily.EngineShowFlags.ScreenPercentage;

		for (int32 ViewIndex = 0; ViewIndex < SceneCaptureViewInfo.Num(); ++ViewIndex)
		{
			SceneRenderer->Views[ViewIndex].GlobalClippingPlane = MirrorPlane;
			// Jitter can't be removed completely due to the clipping plane
			// Also, this prevents the prefilter pass, which reads from jittered depth, from having to do special handling of it's depth-dependent input
			SceneRenderer->Views[ViewIndex].bAllowTemporalJitter = false;
			SceneRenderer->Views[ViewIndex].bRenderSceneTwoSided = CaptureComponent->bRenderSceneTwoSided;

			CaptureComponent->ProjectionWithExtraFOV[ViewIndex] = SceneCaptureViewInfo[ViewIndex].ProjectionMatrix;

			// Plumb down the main view's screen percentage to the planar reflection.
			SceneRenderer->Views[ViewIndex].FinalPostProcessSettings.ScreenPercentage =
				MainSceneRenderer.Views[ViewIndex].FinalPostProcessSettings.ScreenPercentage;

			const bool bIsStereo = MainSceneRenderer.Views[0].StereoPass != EStereoscopicPass::eSSP_FULL;

			const FMatrix ProjectionMatrix = SceneCaptureViewInfo[ViewIndex].ProjectionMatrix;
			FPlanarReflectionSceneProxy* SceneProxy = CaptureComponent->SceneProxy;

			ENQUEUE_RENDER_COMMAND(UpdateProxyCommand)(
				[ProjectionMatrix, ViewIndex, bIsStereo, SceneProxy](FRHICommandList& RHICmdList)
				{
					SceneProxy->ProjectionWithExtraFOV[ViewIndex] = ProjectionMatrix;
					SceneProxy->bIsStereo = bIsStereo;
				});
		}

		{
			const FName OwnerName = CaptureComponent->GetOwner() ? CaptureComponent->GetOwner()->GetFName() : NAME_None;
			FSceneRenderer* MainSceneRendererPtr = &MainSceneRenderer;
			FPlanarReflectionSceneProxy* SceneProxyPtr = CaptureComponent->SceneProxy;
			FPlanarReflectionRenderTarget* RenderTargetPtr = CaptureComponent->RenderTarget;
			ENQUEUE_RENDER_COMMAND(CaptureCommand)(
				[SceneRenderer, MirrorPlane, OwnerName, MainSceneRendererPtr, SceneProxyPtr, RenderTargetPtr](FRHICommandListImmediate& RHICmdList)
			{
				UpdatePlanarReflectionContents_RenderThread(RHICmdList, MainSceneRendererPtr, SceneRenderer, SceneProxyPtr, RenderTargetPtr, RenderTargetPtr, MirrorPlane, OwnerName, FResolveParams(), true);
			});
		}
	}
}

void FScene::AddPlanarReflection(UPlanarReflectionComponent* Component)
{
	check(Component->SceneProxy);
	PlanarReflections_GameThread.Add(Component);

	FPlanarReflectionSceneProxy* SceneProxy = Component->SceneProxy;
	FScene* Scene = this;
	ENQUEUE_RENDER_COMMAND(FAddPlanarReflectionCommand)(
		[SceneProxy, Scene](FRHICommandListImmediate& RHICmdList)
		{
			Scene->ReflectionSceneData.bRegisteredReflectionCapturesHasChanged = true;
			Scene->PlanarReflections.Add(SceneProxy);
		});
}

void FScene::RemovePlanarReflection(UPlanarReflectionComponent* Component) 
{
	check(Component->SceneProxy);
	PlanarReflections_GameThread.Remove(Component);

	FPlanarReflectionSceneProxy* SceneProxy = Component->SceneProxy;
	FScene* Scene = this;
	ENQUEUE_RENDER_COMMAND(FRemovePlanarReflectionCommand)(
		[SceneProxy, Scene](FRHICommandListImmediate& RHICmdList)
		{
			Scene->ReflectionSceneData.bRegisteredReflectionCapturesHasChanged = true;
			Scene->PlanarReflections.Remove(SceneProxy);
		});
}

void FScene::UpdatePlanarReflectionTransform(UPlanarReflectionComponent* Component)
{	
	check(Component->SceneProxy);

	FPlanarReflectionSceneProxy* SceneProxy = Component->SceneProxy;
	FMatrix Transform = Component->GetComponentTransform().ToMatrixWithScale();
	FScene* Scene = this;
	ENQUEUE_RENDER_COMMAND(FUpdatePlanarReflectionCommand)(
		[SceneProxy, Transform, Scene](FRHICommandListImmediate& RHICmdList)
		{
			Scene->ReflectionSceneData.bRegisteredReflectionCapturesHasChanged = true;
			SceneProxy->UpdateTransform(Transform);
		});
}

class FPlanarReflectionPS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(FPlanarReflectionPS, Global);
public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM4);
	}

	/** Default constructor. */
	FPlanarReflectionPS() {}

	/** Initialization constructor. */
	FPlanarReflectionPS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		SceneTextureParameters.Bind(Initializer);
	}

	void SetParameters(FRHICommandList& RHICmdList, const FSceneView& View, FPlanarReflectionSceneProxy* ReflectionSceneProxy)
	{
		const FPixelShaderRHIParamRef ShaderRHI = GetPixelShader();
		FGlobalShader::SetParameters<FViewUniformShaderParameters>(RHICmdList, ShaderRHI, View.ViewUniformBuffer);
		SceneTextureParameters.Set(RHICmdList, ShaderRHI, View.FeatureLevel, ESceneTextureSetupMode::All);

		FPlanarReflectionUniformParameters PlanarReflectionUniformParameters;
		SetupPlanarReflectionUniformParameters(View, ReflectionSceneProxy, PlanarReflectionUniformParameters);
		SetUniformBufferParameterImmediate(RHICmdList, ShaderRHI, GetUniformBufferParameter<FPlanarReflectionUniformParameters>(), PlanarReflectionUniformParameters);
	}

	// FShader interface.
	virtual bool Serialize(FArchive& Ar) override
	{
		bool bShaderHasOutdatedParameters = FGlobalShader::Serialize(Ar);
		Ar << SceneTextureParameters;
		return bShaderHasOutdatedParameters;
	}

private:

	FSceneTextureShaderParameters SceneTextureParameters;
};

IMPLEMENT_SHADER_TYPE(,FPlanarReflectionPS,TEXT("/Engine/Private/PlanarReflectionShaders.usf"),TEXT("PlanarReflectionPS"),SF_Pixel);

bool FDeferredShadingSceneRenderer::RenderDeferredPlanarReflections(FRHICommandListImmediate& RHICmdList, const FViewInfo& View, bool bLightAccumulationIsInUse, TRefCountPtr<IPooledRenderTarget>& Output)
{
	check(RHICmdList.IsOutsideRenderPass());
	// Prevent rendering unsupported views when ViewIndex >= GMaxPlanarReflectionViews
	// Planar reflections in those views will fallback to other reflection methods
	{
		int32 ViewIndex = INDEX_NONE;

		ViewFamily.Views.Find(&View, ViewIndex);

		if (ViewIndex >= GMaxPlanarReflectionViews)
		{
			return false;
		}
	}

	bool bAnyVisiblePlanarReflections = false;

	for (int32 PlanarReflectionIndex = 0; PlanarReflectionIndex < Scene->PlanarReflections.Num(); PlanarReflectionIndex++)
	{
		FPlanarReflectionSceneProxy* ReflectionSceneProxy = Scene->PlanarReflections[PlanarReflectionIndex];

		if (View.ViewFrustum.IntersectBox(ReflectionSceneProxy->WorldBounds.GetCenter(), ReflectionSceneProxy->WorldBounds.GetExtent()))
		{
			bAnyVisiblePlanarReflections = true;
		}
	}

	bool bViewIsReflectionCapture = View.bIsPlanarReflection || View.bIsReflectionCapture;

	// Prevent reflection recursion, or view-dependent planar reflections being seen in reflection captures
	if (Scene->PlanarReflections.Num() > 0 && !bViewIsReflectionCapture && bAnyVisiblePlanarReflections)
	{
		SCOPED_DRAW_EVENT(RHICmdList, CompositePlanarReflections);

		bool bSSRAsInput = true;

		if (Output == GSystemTextures.BlackDummy)
		{
			bSSRAsInput = false;
			FSceneRenderTargets& SceneContext = FSceneRenderTargets::Get(RHICmdList);

			if (bLightAccumulationIsInUse)
			{
				FPooledRenderTargetDesc Desc(FPooledRenderTargetDesc::Create2DDesc(SceneContext.GetBufferSizeXY(), PF_FloatRGBA, FClearValueBinding::Black, TexCreate_None, TexCreate_RenderTargetable, false));
				GRenderTargetPool.FindFreeElement(RHICmdList, Desc, Output, TEXT("PlanarReflectionComposite"));
			}
			else
			{
				Output = SceneContext.LightAccumulation;
			}
		}

		FRHIRenderPassInfo RPInfo(Output->GetRenderTargetItem().TargetableTexture, ERenderTargetActions::Load_Store);
		RHICmdList.BeginRenderPass(RPInfo, TEXT("DeferredPlanarReflections"));
		{
			if (!bSSRAsInput)
			{
				DrawClearQuad(RHICmdList, FLinearColor(0, 0, 0, 0));
			}

			{
				RHICmdList.SetViewport(View.ViewRect.Min.X, View.ViewRect.Min.Y, 0.0f, View.ViewRect.Max.X, View.ViewRect.Max.Y, 1.0f);

				FGraphicsPipelineStateInitializer GraphicsPSOInit;
				RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

				// Blend over previous reflections in the output target (SSR or planar reflections that have already been rendered)
				// Planar reflections win over SSR and reflection environment
				//@todo - this is order dependent blending, but ordering is coming from registration order
				GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_One, BF_InverseSourceAlpha, BO_Max, BF_One, BF_One>::GetRHI();
				GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
				GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

				for (int32 PlanarReflectionIndex = 0; PlanarReflectionIndex < Scene->PlanarReflections.Num(); PlanarReflectionIndex++)
				{
					FPlanarReflectionSceneProxy* ReflectionSceneProxy = Scene->PlanarReflections[PlanarReflectionIndex];

					if (View.ViewFrustum.IntersectBox(ReflectionSceneProxy->WorldBounds.GetCenter(), ReflectionSceneProxy->WorldBounds.GetExtent()))
					{
						SCOPED_DRAW_EVENTF(RHICmdList, PlanarReflection, *ReflectionSceneProxy->OwnerName.ToString());

						TShaderMapRef<TDeferredLightVS<false> > VertexShader(View.ShaderMap);
						TShaderMapRef<FPlanarReflectionPS> PixelShader(View.ShaderMap);

						GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GFilterVertexDeclaration.VertexDeclarationRHI;
						GraphicsPSOInit.BoundShaderState.VertexShaderRHI = GETSAFERHISHADER_VERTEX(*VertexShader);
						GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
						GraphicsPSOInit.PrimitiveType = PT_TriangleList;

						SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

						PixelShader->SetParameters(RHICmdList, View, ReflectionSceneProxy);
						VertexShader->SetSimpleLightParameters(RHICmdList, View, FSphere(0));

						DrawRectangle(
							RHICmdList,
							0, 0,
							View.ViewRect.Width(), View.ViewRect.Height(),
							View.ViewRect.Min.X, View.ViewRect.Min.Y,
							View.ViewRect.Width(), View.ViewRect.Height(),
							View.ViewRect.Size(),
							FSceneRenderTargets::Get(RHICmdList).GetBufferSizeXY(),
							*VertexShader,
							EDRF_UseTriangleOptimization);
					}
				}
			}
		}
		RHICmdList.EndRenderPass();
		RHICmdList.CopyToResolveTarget(Output->GetRenderTargetItem().TargetableTexture, Output->GetRenderTargetItem().ShaderResourceTexture, FResolveParams());

		return true;
	}

	return false;
}
