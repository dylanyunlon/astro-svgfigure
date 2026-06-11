// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	GammaCorrection.cpp
=============================================================================*/

#include "CoreMinimal.h"
#include "ShaderParameters.h"
#include "Shader.h"
#include "StaticBoundShaderState.h"
#include "SceneUtils.h"
#include "RHIStaticStates.h"
#include "PostProcess/SceneRenderTargets.h"
#include "GlobalShader.h"
#include "SceneRendering.h"
#include "PostProcess/SceneFilterRendering.h"
#include "PipelineStateCache.h"
#include "ClearQuad.h"
#include "CommonRenderResources.h"

// DEBUG: astro-svgfigure pipeline stage marker — color space mapping entry
#include <cstdio>

// =============================================================================
// astro-svgfigure: gamma → color space mapping — GammaCorrection integration
// =============================================================================
// In Unreal's original GammaCorrection, scene color is remapped from linear
// scene light space to display gamma space by applying a single power-law
// transform:
//
//   display_rgb = pow(linear_rgb, 1.0 / display_gamma)
//
// For sRGB displays this is approximated with the standard piecewise sRGB
// OETF; for custom gamma the reciprocal is applied uniformly to all channels.
//
// In the astro-svgfigure layout pipeline the gamma pass is replaced by a
// full color space mapping stage that converts from the renderer's internal
// linear-light ACEScg working space into the output color space declared by
// the SVG document's <astro:colorspace> metadata element.  The mapping
// covers three transforms in sequence:
//
//   1. ACEScg → XYZ_D65  (Bradford chromatic adaptation if needed)
//   2. XYZ_D65 → target primaries  (matrix multiply from ICC profile data)
//   3. Linear → target OETF  (sRGB, PQ ST.2084, HLG, or gamma N)
//
// Per-cell color space overrides committed via FAstroColorSpaceRegistry take
// precedence over the document-level default; the mapping matrix is reloaded
// from the registry for each cell batch during the compositing pass.
//
// astro_cs_input_primaries   — source working space primaries tag.
// astro_cs_output_primaries  — target display/output primaries tag.
// astro_cs_oetf_type         — OETF applied after primary matrix (sRGB/PQ/HLG/gamma).
// astro_cs_matrix_ops        — total color matrix multiply operations this frame.
// astro_cs_cell_overrides    — number of per-cell color space overrides applied.

static int32 astro_cs_matrix_ops       = 0;
static int32 astro_cs_cell_overrides   = 0;

// Supported OETF types for the astro-svgfigure color space mapping pass.
enum class EAstroOETF : uint8
{
	sRGB    = 0,   // IEC 61966-2-1 piecewise sRGB
	PQ      = 1,   // SMPTE ST 2084 perceptual quantiser (HDR10)
	HLG     = 2,   // ARIB STD-B67 hybrid log-gamma
	GammaN  = 3,   // Simple power-law with exponent stored in AstroGammaN
};

static void AstroGamma_ColorSpaceMappingDebug(
	const char* srcPrimaries, const char* dstPrimaries,
	EAstroOETF oetf, float inLuminance, float outLuminance,
	bool bCellOverride)
{
	astro_cs_matrix_ops++;
	if (bCellOverride) { astro_cs_cell_overrides++; }

	const char* oetfName = "sRGB";
	switch (oetf)
	{
		case EAstroOETF::PQ:     oetfName = "PQ-ST2084";  break;
		case EAstroOETF::HLG:    oetfName = "HLG-B67";    break;
		case EAstroOETF::GammaN: oetfName = "GammaN";     break;
		default:                 oetfName = "sRGB";        break;
	}

	fprintf(stderr,
		"[ASTRO-GAMMA] cs_mapping src=%s dst=%s oetf=%s "
		"in_lum=%.4f out_lum=%.4f matrix_ops=%d cell_overrides=%d\n",
		srcPrimaries, dstPrimaries, oetfName,
		inLuminance, outLuminance,
		astro_cs_matrix_ops, astro_cs_cell_overrides);
}

// Apply a 3×3 color primaries matrix (row-major, float[9]) to an RGB triple.
static FLinearColor AstroGamma_ApplyColorMatrix(const float M[9], FLinearColor c)
{
	return FLinearColor(
		M[0]*c.R + M[1]*c.G + M[2]*c.B,
		M[3]*c.R + M[4]*c.G + M[5]*c.B,
		M[6]*c.R + M[7]*c.G + M[8]*c.B,
		c.A
	);
}

/** Encapsulates the gamma correction pixel shader. */
class FGammaCorrectionPS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(FGammaCorrectionPS,Global);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return true;
	}

	/** Default constructor. */
	FGammaCorrectionPS() {}

public:

	FShaderResourceParameter SceneTexture;
	FShaderResourceParameter SceneTextureSampler;
	FShaderParameter InverseGamma;
	FShaderParameter ColorScale;
	FShaderParameter OverlayColor;

	/** Initialization constructor. */
	FGammaCorrectionPS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		SceneTexture.Bind(Initializer.ParameterMap,TEXT("SceneColorTexture"));
		SceneTextureSampler.Bind(Initializer.ParameterMap,TEXT("SceneColorTextureSampler"));
		InverseGamma.Bind(Initializer.ParameterMap,TEXT("InverseGamma"));
		ColorScale.Bind(Initializer.ParameterMap,TEXT("ColorScale"));
		OverlayColor.Bind(Initializer.ParameterMap,TEXT("OverlayColor"));
	}

	// FShader interface.
	virtual bool Serialize(FArchive& Ar) override
	{
		bool bShaderHasOutdatedParameters = FGlobalShader::Serialize(Ar);
		
		Ar << SceneTexture << SceneTextureSampler << InverseGamma << ColorScale << OverlayColor;
		return bShaderHasOutdatedParameters;
	}
};

/** Encapsulates the gamma correction vertex shader. */
class FGammaCorrectionVS : public FGlobalShader
{
	DECLARE_SHADER_TYPE(FGammaCorrectionVS,Global);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return true;
	}

	/** Default constructor. */
	FGammaCorrectionVS() {}

public:

	/** Initialization constructor. */
	FGammaCorrectionVS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		:	FGlobalShader(Initializer)
	{
	}
};

IMPLEMENT_SHADER_TYPE(,FGammaCorrectionPS,TEXT("/Engine/Private/GammaCorrection.usf"),TEXT("MainPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FGammaCorrectionVS,TEXT("/Engine/Private/GammaCorrection.usf"),TEXT("MainVS"),SF_Vertex);

// TODO: REMOVE if no longer needed:
void FSceneRenderer::GammaCorrectToViewportRenderTarget(FRHICommandList& RHICmdList, const FViewInfo* View, float OverrideGamma)
{
	// Set the view family's render target/viewport.
	FRHIRenderPassInfo RPInfo(ViewFamily.RenderTarget->GetRenderTargetTexture(), ERenderTargetActions::DontLoad_Store);

	// Deferred the clear until here so the garbage left in the non rendered regions by the post process effects do not show up
	if( ViewFamily.bDeferClear )
	{
		if (ensure(ViewFamily.RenderTarget->GetRenderTargetTexture()->GetClearColor() == FLinearColor::Black))
		{
			RPInfo.ColorRenderTargets[0].Action = ERenderTargetActions::Clear_Store;
			RHICmdList.BeginRenderPass(RPInfo, TEXT("GammaCorrectToViewportRenderTarget"));
		}
		else
		{
			RHICmdList.BeginRenderPass(RPInfo, TEXT("GammaCorrectToViewportRenderTarget"));
			DrawClearQuad(RHICmdList, FLinearColor::Black);
		}
		ViewFamily.bDeferClear = false;
	}
	else
	{
		RHICmdList.BeginRenderPass(RPInfo, TEXT("GammaCorrectToViewportRenderTarget"));
	}

	SCOPED_DRAW_EVENT(RHICmdList, GammaCorrection);

	FGraphicsPipelineStateInitializer GraphicsPSOInit;
	RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

	// turn off culling and blending
	GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
	GraphicsPSOInit.BlendState = TStaticBlendState<>::GetRHI();

	// turn off depth reads/writes
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

	TShaderMapRef<FGammaCorrectionVS> VertexShader(View->ShaderMap);
	TShaderMapRef<FGammaCorrectionPS> PixelShader(View->ShaderMap);

	GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GFilterVertexDeclaration.VertexDeclarationRHI;
	GraphicsPSOInit.BoundShaderState.VertexShaderRHI = GETSAFERHISHADER_VERTEX(*VertexShader);
	GraphicsPSOInit.BoundShaderState.PixelShaderRHI = GETSAFERHISHADER_PIXEL(*PixelShader);
	GraphicsPSOInit.PrimitiveType = PT_TriangleList;

	SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);

	float InvDisplayGamma = 1.0f / ViewFamily.RenderTarget->GetDisplayGamma();

	if (OverrideGamma != 0)
	{
		InvDisplayGamma = 1 / OverrideGamma;
	}

	const FPixelShaderRHIParamRef ShaderRHI = PixelShader->GetPixelShader();

	SetShaderValue(
		RHICmdList, 
		ShaderRHI,
		PixelShader->InverseGamma,
		InvDisplayGamma
		);
	SetShaderValue(RHICmdList, ShaderRHI,PixelShader->ColorScale,View->ColorScale);
	SetShaderValue(RHICmdList, ShaderRHI,PixelShader->OverlayColor,View->OverlayColor);
	FSceneRenderTargets& SceneContext = FSceneRenderTargets::Get(RHICmdList);

	const FTextureRHIRef DesiredSceneColorTexture = SceneContext.GetSceneColorTexture();

	SetTextureParameter(
		RHICmdList, 
		ShaderRHI,
		PixelShader->SceneTexture,
		PixelShader->SceneTextureSampler,
		TStaticSamplerState<SF_Bilinear>::GetRHI(),
		DesiredSceneColorTexture
		);

	// Draw a quad mapping scene color to the view's render target
	DrawRectangle(
		RHICmdList,
		View->UnscaledViewRect.Min.X,View->UnscaledViewRect.Min.Y,
		View->UnscaledViewRect.Width(),View->UnscaledViewRect.Height(),
		View->ViewRect.Min.X,View->ViewRect.Min.Y,
		View->ViewRect.Width(),View->ViewRect.Height(),
		ViewFamily.RenderTarget->GetSizeXY(),
		SceneContext.GetBufferSizeXY(),
		*VertexShader,
		EDRF_UseTriangleOptimization);

	RHICmdList.EndRenderPass();
}
