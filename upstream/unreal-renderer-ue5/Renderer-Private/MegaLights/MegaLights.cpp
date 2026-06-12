// Copyright Epic Games, Inc. All Rights Reserved.

#include "MegaLights.h"
#include "MegaLightsInternal.h"
#include "PixelShaderUtils.h"
#include "BasePassRendering.h"
#include "VolumetricFogShared.h"
#include "Shadows/ShadowSceneRenderer.h"
#include "HairStrands/HairStrandsData.h"
#include "StochasticLighting/StochasticLighting.h"
#include "Froxel/FroxelGridUtils.h"
#include "FirstPersonSceneExtension.h"
#include "SceneCore.h"
#include "ScenePrivate.h"
#include "SceneViewState.h"
#include "DeferredShadingRenderer.h"
#include "LogRenderer.h"

static TAutoConsoleVariable<bool> CVarMegaLightsSupported(
	TEXT("r.MegaLights.Supported"),
	true,
	TEXT("Whether MegaLights is supported at all for the project, regardless of platform. This can be used to avoid compiling MegaLights shaders."),
	ECVF_ReadOnly | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsProjectSetting(
	TEXT("r.MegaLights.EnableForProject"),
	0,
	TEXT("Whether to use MegaLights by default, but this can still be overridden by Post Process Volumes, or disabled per-light. MegaLights uses stochastic sampling to render many shadow casting lights efficiently, with a consistent low GPU cost. MegaLights requires Hardware Ray Tracing, and does not support Directional Lights. Experimental feature."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsAllowed(
	TEXT("r.MegaLights.Allowed"),
	1,
	TEXT("Whether the MegaLights feature is allowed by scalability and device profiles."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsLightingDataFormat(
	TEXT("r.MegaLights.LightingDataFormat"),
	0,
	TEXT("Data format for surfaces storing lighting information (e.g. radiance, irradiance).\n")
	TEXT("0 - Float_R11G11B10 (fast default)\n")
	TEXT("1 - Float16_RGBA (slow but higher precision, mostly for testing)\n")
	TEXT("2 - Float32_RGBA (reference for testing)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDownsampleMode(
	TEXT("r.MegaLights.DownsampleMode"),
	2,
	TEXT("Downsample mode from the main viewport to sample and trace rays. Increases performance, but reduces quality.\n")
	TEXT("0 - Disabled (1x1)\n")
	TEXT("1 - Checkerboard (2x1)\n")
	TEXT("2 - Half-resolution (2x2)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsNumSamplesPerPixel(
	TEXT("r.MegaLights.NumSamplesPerPixel"),
	4,
	TEXT("Number of samples per downsampled pixel for the opaque lighting pass.\n")
	TEXT("Final number of samples depends also on the downsampling mode.\n")
	TEXT("Supported values: 1, 2 and 4."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsMinSampleWeight(
	TEXT("r.MegaLights.MinSampleWeight"),
	0.001f,
	TEXT("Determines minimal sample influence on final pixels. Used to skip samples which would have minimal impact to the final image even if light is fully visible."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsLightAttenuationFalloff(
	TEXT("r.MegaLights.LightAttenuationFalloff"),
	0.18f,
	TEXT("Average base color luminance used for early light culling. Culling is based on potential light energy contribution estimated using light color, falloff and set base color luminance. ")
	TEXT("Lower values will make light sampling pass faster, but can cause some lights to be cutoff too early (especially specular on smooth surfaces).\n")
	TEXT("1 is conservative (assumes fully white diffuse and light perpendicular to the surface).\n")
	TEXT("0 will disable any early culling."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsGuideByHistory(
	TEXT("r.MegaLights.GuideByHistory"),
	true,
	TEXT("Whether to reduce sampling chance for lights which were hidden last frame. Reduces noise in areas where bright lights are shadowed."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsWaveOps(
	TEXT("r.MegaLights.WaveOps"),
	true,
	TEXT("Whether to use wave ops. Useful for debugging."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFastClear(
	TEXT("r.MegaLights.FastClear"),
	false,
	TEXT("Whether to skip processing of empty tiles and only clear borders for spatial and temporal filtering. ")
	TEXT("FastClear is a bit slower when all tiles are valid, so it's disabled by default for opaque pixels."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsHairStrandsFastClear(
	TEXT("r.MegaLights.HairStrands.FastClear"),
	true,
	TEXT("Whether to skip processing of empty tiles and only clear borders for spatial and temporal filtering."),
	ECVF_RenderThreadSafe
);

// Keep in sync with EMegaLightsDebugMode
static TAutoConsoleVariable<int32> CVarMegaLightsDebug(
	TEXT("r.MegaLights.Debug"),
	0,
	TEXT("Whether to enabled debug mode, which prints various extra debug information from shaders.\n")
	TEXT("0 - Disable\n")
	TEXT("1 - Opaque\n")
	TEXT("2 - Volume\n")
	TEXT("3 - Translucency Volume\n")
	TEXT("4 - Hair Strands\n")
	TEXT("5 - Front Layer Translucency"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugCursorX(
	TEXT("r.MegaLights.Debug.CursorX"),
	-1,
	TEXT("Override default debug visualization cursor position."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugCursorY(
	TEXT("r.MegaLights.Debug.CursorY"),
	-1,
	TEXT("Override default debug visualization cursor position."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugVolumeSliceIndex(
	TEXT("r.MegaLights.Debug.VolumeSliceIndex"),
	-1,
	TEXT("Which volume slice to debug."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugLightId(
	TEXT("r.MegaLights.Debug.LightId"),
	-1,
	TEXT("Which light to show debug info for. When set to -1, uses the currently selected light in editor."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugVisualizeLight(
	TEXT("r.MegaLights.Debug.VisualizeLight"),
	0,
	TEXT("Whether to visualize selected light. Useful to find in in the level."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDebugVisualizeTraces(
	TEXT("r.MegaLights.Debug.VisualizeTraces"),
	1,
	TEXT("How to draw traces for the selected pixel.")
	TEXT("0 - Disabled\n")
	TEXT("1 - Draw traced rays\n")
	TEXT("2 - Draw samples"),
	ECVF_RenderThreadSafe
);

int32 GMegaLightsReset = 0;
FAutoConsoleVariableRef CVarMegaLightsReset(
	TEXT("r.MegaLights.Reset"),
	GMegaLightsReset,
	TEXT("Reset history for debugging."),
	ECVF_RenderThreadSafe
);

int32 GMegaLightsResetEveryNthFrame = 0;
	FAutoConsoleVariableRef CVarMegaLightsResetEveryNthFrame(
	TEXT("r.MegaLights.ResetEveryNthFrame"),
		GMegaLightsResetEveryNthFrame,
	TEXT("Reset history every Nth frame for debugging."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsFixedStateFrameIndex(
	TEXT("r.MegaLights.FixedStateFrameIndex"),
	-1,
	TEXT("Whether to override View.StateFrameIndex for debugging."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTexturedRectLights(
	TEXT("r.MegaLights.TexturedRectLights"),
	1,
	TEXT("Whether to support textured rect lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsLightFunctions(
	TEXT("r.MegaLights.LightFunctions"),
	1,
	TEXT("Whether to support light functions."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightLightingChannels(
	TEXT("r.MegaLights.LightingChannels"),
	1,
	TEXT("Whether to enable lighting channels to block shadowing"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsIESProfiles(
	TEXT("r.MegaLights.IESProfiles"),
	1,
	TEXT("Whether to support IES profiles on lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsTransmission(
	TEXT("r.MegaLights.Transmission"),
	true,
	TEXT("Whether to support subsurface transmission for lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsTransmissionSampleWeight(
	TEXT("r.MegaLights.Transmission.SampleWeight"),
	0.1f,
	TEXT("Assumed transmission value when evaluating the BSDF for the sample weight, before the transmission is actually traced and evaluated. Can be used to tweak sampling noise of surfaces with transmission."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDirectionalLights(
	TEXT("r.MegaLights.DirectionalLights"),
	0,
	TEXT("Whether to support directional lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolume(
	TEXT("r.MegaLights.Volume"),
	1,
	TEXT("Whether to enable a translucency volume used for Volumetric Fog and Volume Lit Translucency."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeUnified(
	TEXT("r.MegaLights.Volume.Unified"),
	1,
	TEXT("Whether to reuse sampling / tracing for volumetric fog and translucency volume."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarMegaLightsVolumeDepthDistributionScale(
	TEXT("r.MegaLights.Volume.DepthDistributionScale"),
	32.0f,
	TEXT("Scales the slice depth distribution."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeGridPixelSize(
	TEXT("r.MegaLights.Volume.GridPixelSize"),
	8,
	TEXT("XY size of a cell in the voxel grid, in pixels."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeGridSizeZ(
	TEXT("r.MegaLights.Volume.GridSizeZ"),
	128,
	TEXT("How many cells in the voxel grid to use in z."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeDownsampleMode(
	TEXT("r.MegaLights.Volume.DownsampleMode"),
	2,
	TEXT("Downsample mode applied for volume (Volumetric Fog and Lit Translucency) to sample and trace rays. Increases performance, but reduces quality.\n")
	TEXT("0 - Disabled (1x1x1)\n")
	TEXT("1 - Reserved for a future mode\n")
	TEXT("2 - Half-resolution (2x2x2)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeHZBOcclusionTest(
	TEXT("r.MegaLights.Volume.HZBOcclusionTest"),
	1,
	TEXT("Whether to skip computation for cells occluded by HZB."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeNumSamplesPerVoxel(
	TEXT("r.MegaLights.Volume.NumSamplesPerVoxel"),
	2,
	TEXT("Number of samples per downsampled voxel for the volume lighting pass.\n")
	TEXT("Final number of samples depends also on the downsampling mode.\n")
	TEXT("Supported values: 2 and 4."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsVolumeMinSampleWeight(
	TEXT("r.MegaLights.Volume.MinSampleWeight"),
	0.1f,
	TEXT("Determines minimal sample influence on lighting cached in a volume. Used to skip samples which would have minimal impact to the final image even if light is fully visible."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeGuideByHistory(
	TEXT("r.MegaLights.Volume.GuideByHistory"),
	1,
	TEXT("Whether to reduce sampling chance for lights which were hidden last frame. Reduces noise in areas where bright lights are shadowed.\n")
	TEXT("0 - disabled\n")
	TEXT("1 - more rays towards visible lights"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolume(
	TEXT("r.MegaLights.TranslucencyVolume"),
	1,
	TEXT("Whether to enable Lit Translucency Volume."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolumeDownsampleFactor(
	TEXT("r.MegaLights.TranslucencyVolume.DownsampleFactor"),
	2,
	TEXT("Downsample factor applied to Translucency Lighting Volume resolution. Affects the resolution at which rays are traced."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolumeNumSamplesPerVoxel(
	TEXT("r.MegaLights.TranslucencyVolume.NumSamplesPerVoxel"),
	2,
	TEXT("Number of samples (shadow rays) per half-res voxel. Supported values: 2 and 4."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsTranslucencyVolumeMinSampleWeight(
	TEXT("r.MegaLights.TranslucencyVolume.MinSampleWeight"),
	0.1f,
	TEXT("Determines minimal sample influence on lighting cached in a volume. Used to skip samples which would have minimal impact to the final image even if light is fully visible."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolumeSpatial(
	TEXT("r.MegaLights.TranslucencyVolume.Spatial"),
	1,
	TEXT("Whether to run a spatial filter when updating the translucency volume."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolumeTemporal(
	TEXT("r.MegaLights.TranslucencyVolume.Temporal"),
	1,
	TEXT("Whether to use temporal accumulation when updating the translucency volume."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTranslucencyVolumeGuideByHistory(
	TEXT("r.MegaLights.TranslucencyVolume.GuideByHistory"),
	1,
	TEXT("Whether to reduce sampling chance for lights which were hidden last frame. Reduces noise in areas where bright lights are shadowed.\n")
	TEXT("0 - disabled\n")
	TEXT("1 - more rays towards visible lights"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

// Rendering project setting
int32 GMegaLightsDefaultShadowMethod = 0;
FAutoConsoleVariableRef CMegaLightsDefaultShadowMethod(
	TEXT("r.MegaLights.DefaultShadowMethod"),
	GMegaLightsDefaultShadowMethod,
	TEXT("The default shadowing method for MegaLights, unless over-ridden on the light component.\n")
	TEXT("0 - Ray Tracing. Preferred method, which guarantees fixed MegaLights cost and correct area shadows, but is dependent on the BVH representation quality.\n")
	TEXT("1 - Virtual Shadow Maps. Has a significant per light cost, but can cast shadows directly from the Nanite geometry using rasterization."),
	ECVF_RenderThreadSafe
);

// Whether the user setting should be respected based on the current scalability level
static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencyAllowed(
	TEXT("r.MegaLights.FrontLayerTranslucency.Allow"),
	false,
	TEXT("Whether the current scalability level allows a dedicated MegaLights pass on the frontmost layer of Translucent Surfaces.  Other layers will use the lower quality Translucency Lighting Volume."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

// Note: Driven by URendererSettings
static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencyEnabledForProject(
	TEXT("r.MegaLights.FrontLayerTranslucency.EnableForProject"),
	true,
	TEXT("Whether to use a dedicated MegaLights pass on the frontmost layer of Translucent Surfaces.  Other layers will use the lower quality Translucency Lighting Volume.\n")
	TEXT("This setting can be overriden by post process volumes."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsFrontLayerTranslucencyDebugOverrideEnable(
	TEXT("r.MegaLights.FrontLayerTranslucency.Debug.OverrideEnable"),
	-1,
	TEXT("-1 = Respect scalability / project settings / post process volume.\n")
	TEXT(" 0 = Force disable front layer translucency.\n")
	TEXT(" 1 = Force enable front layer translucency."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencyFastClear(
	TEXT("r.MegaLights.FrontLayerTranslucency.FastClear"),
	true,
	TEXT("Whether to skip processing of empty tiles and only clear borders for spatial and temporal filtering."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencySpecularOnly(
	TEXT("r.MegaLights.FrontLayerTranslucency.SpecularOnly"),
	false,
	TEXT("Whether the front layer translucency pass should only calculate specular lighting.\n")
	TEXT("When this is enabled, diffuse lighting is calculated using the translucency lighting volume.\n")
	TEXT("Can be used as a scalability option since it significantly reduces the number rays traced and pixels shaded."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencySoftShadows(
	TEXT("r.MegaLights.FrontLayerTranslucency.SoftShadows"),
	true,
	TEXT("Whether the front layer translucency pass should support soft shadows from area lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencyTexturedRectLights(
	TEXT("r.MegaLights.FrontLayerTranslucency.TexturedRectLights"),
	true,
	TEXT("Whether the front layer translucency pass should support textured rect lights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsEnableHairStrands(
	TEXT("r.MegaLights.HairStrands"),
	1,
	TEXT("Wheter to enable hair strands support for MegaLights."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHairStrandsDownsampleMode(
	TEXT("r.MegaLights.HairStrands.DownsampleMode"),
	0,
	TEXT("Downsample mode from the main viewport to sample and trace rays for hair strands. Increases performance, but reduces quality.\n")
	TEXT("0 - Disabled (1x1)\n")
	TEXT("1 - Checkerboard (2x1)\n")
	TEXT("2 - Half-resolution (2x2)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHairStrandsNumSamplesPerPixel(
	TEXT("r.MegaLights.HairStrands.NumSamplesPerPixel"),
	4,
	TEXT("Number of samples per downsampled pixel for the hair strands lighting pass.\n")
	TEXT("Final number of samples depends also on the downsampling mode.\n")
	TEXT("Supported values: 1, 2 and 4."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHairStrandsSubPixelShading(
	TEXT("r.MegaLights.HairStrands.SubPixelShading"),
	0,
	TEXT("Shader all sub-pixel data for better quality (add extra cost)."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsReferenceOffsetToStateFrameIndex(
	TEXT("r.MegaLights.Reference.OffsetToStateFrameIndex"),
	0,
	TEXT("Offset to add to View.StateFrameIndex."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsReferenceShadingPassCount(
	TEXT("r.MegaLights.Reference.NumShadingPass"),
	1,
	TEXT("Number of pass for shading (to generate references at the cost of performance when pass count is > 1)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsReferenceDebuggedPassIndex(
	TEXT("r.MegaLights.Reference.DebuggedPassIndex"),
	-1,
	TEXT("When r.MegaLights.Debug is activated, the pass index to print debug info from.\n.")
	TEXT("Use negative value to index in reverse order.\n.")
	TEXT("Default is -1 meaning the last pass.\n."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVSMMarkPages(
	TEXT("r.MegaLights.VSM.MarkPages"),
	1,
	TEXT("When enabled, MegaLights will mark Virtual Shadow Map pages for required samples directly.\n")
	TEXT("Otherwise any light using MegaLights VSM will mark all pages that conservatively might be required."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsAsyncComputeVolume(
	TEXT("r.MegaLights.AsyncCompute.Volume"),
	0,
	TEXT("Whether to run volume and TLV passes on async compute."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsAsyncComputeGenerateSamples(
	TEXT("r.MegaLights.AsyncCompute.GenerateSamples"),
	0,
	TEXT("Whether to run light sample generation passes on async compute."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

extern int32 GUseTranslucencyLightingVolumes;
extern int32 GMegaLightsVisualizeLightComplexityDump;

namespace MegaLights
{
	constexpr int32 TileSize = TILE_SIZE;
	constexpr int32 VisibleLightHashTileSize = VISIBLE_LIGHT_HASH_TILE_SIZE;

	// Must match MegaLightsVisibility.ush 
	constexpr int32 VisibleLightHashSize = 4;
	constexpr int32 HiddenLightHashSize = 2;

	bool ShouldCompileShaders(EShaderPlatform ShaderPlatform)
	{
		return ShouldCompileMegaLightsShaders(ShaderPlatform);
	}

	bool IsRequested(const FSceneViewFamily& ViewFamily)
	{
		return ViewFamily.Views[0]->FinalPostProcessSettings.bMegaLights
			&& CVarMegaLightsAllowed.GetValueOnRenderThread() != 0
			&& ViewFamily.EngineShowFlags.Lighting
			&& ViewFamily.EngineShowFlags.DirectLighting
			&& ViewFamily.EngineShowFlags.MegaLights
			&& ShouldCompileShaders(ViewFamily.GetShaderPlatform());
	}

	bool HasRequiredTracingData(const FSceneViewFamily& ViewFamily)
	{
		return IsHardwareRayTracingSupported(ViewFamily) || IsSoftwareRayTracingSupported(ViewFamily);
	}

	bool IsEnabled(const FSceneViewFamily& ViewFamily)
	{
		return IsRequested(ViewFamily) && HasRequiredTracingData(ViewFamily);
	}

	bool IsEnabledInProject(EShaderPlatform ShaderPlatform)
	{
		return ShouldCompileMegaLightsShaders(ShaderPlatform)
			&& CVarMegaLightsAllowed.GetValueOnRenderThread() != 0
			&& CVarMegaLightsProjectSetting.GetValueOnRenderThread() != 0;
	}

	EPixelFormat GetLightingDataFormat()
	{
		if (CVarMegaLightsLightingDataFormat.GetValueOnRenderThread() == 2)
		{
			return PF_A32B32G32R32F;
		}
		else if (CVarMegaLightsLightingDataFormat.GetValueOnRenderThread() == 1)
		{
			return PF_FloatRGBA;
		}
		else
		{
			return PF_FloatR11G11B10;
		}
	}

	float GetTransmissionSampleWeight()
	{
		if (CVarMegaLightsTransmission.GetValueOnRenderThread())
		{
			return FMath::Clamp(CVarMegaLightsTransmissionSampleWeight.GetValueOnRenderThread(), 0.0f, 1.0f);
		}

		return 0.0f;
	}

	uint32 GetSampleMargin()
	{
		// #ml_todo: should be calculated based on DownsampleFactor / Volume.DownsampleFactor
		return 3;
	}

	bool UseVolume()
	{
		return CVarMegaLightsVolume.GetValueOnRenderThread() != 0;
	}

	bool UseTranslucencyVolume()
	{
		return CVarMegaLightsTranslucencyVolume.GetValueOnRenderThread() != 0;
	}

	bool IsTranslucencyVolumeSpatialFilterEnabled()
	{
		return CVarMegaLightsTranslucencyVolumeSpatial.GetValueOnRenderThread() != 0;
	}

	bool IsTranslucencyVolumeTemporalFilterEnabled()
	{
		return CVarMegaLightsTranslucencyVolumeTemporal.GetValueOnRenderThread() != 0;
	}

	bool UseFrontLayerTranslucencyDirectLighting(const FViewInfo& View)
	{
		if (!MegaLights::IsEnabled(*View.Family))
		{
			return false;
		}

		if (CVarMegaLightsFrontLayerTranslucencyDebugOverrideEnable.GetValueOnRenderThread() >= 0)
		{
			return CVarMegaLightsFrontLayerTranslucencyDebugOverrideEnable.GetValueOnRenderThread() != 0;
		}

		return CVarMegaLightsFrontLayerTranslucencyAllowed.GetValueOnRenderThread() && View.FinalPostProcessSettings.MegaLightsFrontLayerTranslucency;
	}

	bool IsHairStrandsEnabled(const FViewInfo& View)
	{
		return HairStrands::HasViewHairStrandsData(View) && CVarMegaLightsEnableHairStrands.GetValueOnRenderThread() > 0;
	}

	bool IsMarkingVSMPages()
	{
		return CVarMegaLightsVSMMarkPages.GetValueOnRenderThread() != 0;
	}

	bool IsUsingLightFunctions(const FSceneViewFamily& ViewFamily)
	{
		return IsEnabled(ViewFamily) && CVarMegaLightsLightFunctions.GetValueOnRenderThread() != 0;
	}

	bool IsUsingLightingChannels()
	{
		return CVarMegaLightLightingChannels.GetValueOnRenderThread() != 0;
	}

	EMegaLightsMode GetMegaLightsMode(const FSceneViewFamily& ViewFamily, uint8 LightType, bool bLightAllowsMegaLights, TEnumAsByte<EMegaLightsShadowMethod::Type> ShadowMethod)
	{
		if ((LightType != LightType_Directional || CVarMegaLightsDirectionalLights.GetValueOnRenderThread())
			&& IsEnabled(ViewFamily) 
			&& bLightAllowsMegaLights)
		{
			// Resolve  default
			if (ShadowMethod == EMegaLightsShadowMethod::Default)
			{
				if (GMegaLightsDefaultShadowMethod == 1)
				{
					ShadowMethod = EMegaLightsShadowMethod::VirtualShadowMap;
				}
				else
				{
					ShadowMethod = EMegaLightsShadowMethod::RayTracing;
				}
			}

			const bool bUseVSM = ShadowMethod == EMegaLightsShadowMethod::VirtualShadowMap;

			if (bUseVSM)
			{
				return EMegaLightsMode::EnabledVSM;
			}
			// Just check first view, assuming the ray tracing flag is the same for all views.  See comment in the ShouldRenderRayTracingEffect function that accepts a ViewFamily.
			else if (ViewFamily.Views[0]->IsRayTracingAllowedForView())
			{
				return EMegaLightsMode::EnabledRT;
			}
		}

		return EMegaLightsMode::Disabled;
	}

	bool ShouldCompileShadersForReferenceMode(EShaderPlatform InPlatform, bool bHasEditorOnlyData)
	{
		// Only compile reference mode on PC platform
		return IsPCPlatform(InPlatform) && bHasEditorOnlyData;
	}

	uint32 GetReferenceShadingPassCount(EShaderPlatform InPlatform)
	{
		return ShouldCompileShadersForReferenceMode(InPlatform, WITH_EDITORONLY_DATA) ? (uint32)FMath::Clamp<int32>(CVarMegaLightsReferenceShadingPassCount.GetValueOnAnyThread(), 1, 10*1024) : 1u;
	}

	uint32 GetStateFrameIndex(FSceneViewState* ViewState, EShaderPlatform InPlatform)
	{
		uint32 StateFrameIndex = ViewState ? ViewState->GetFrameIndex() : 0;

		if (CVarMegaLightsFixedStateFrameIndex.GetValueOnRenderThread() >= 0)
		{
			StateFrameIndex = CVarMegaLightsFixedStateFrameIndex.GetValueOnRenderThread();
		}

		if (StochasticLighting::IsStateFrameIndexOverridden())
		{
			StateFrameIndex = StochasticLighting::GetStateFrameIndex(ViewState);
		}

		if (CVarMegaLightsReferenceOffsetToStateFrameIndex.GetValueOnRenderThread() > 0)
		{
			StateFrameIndex += CVarMegaLightsReferenceOffsetToStateFrameIndex.GetValueOnRenderThread();
		}

		//In case we accumulate we account for this in the state frame index to get the same property out of the BlueNoise.
		StateFrameIndex *= GetReferenceShadingPassCount(InPlatform);

		return StateFrameIndex;
	}

	FIntPoint GetDownsampleFactorXY(EMegaLightsInput InputType, EShaderPlatform ShaderPlatform)
	{
		uint32 DownsampleMode = 0;

		if (InputType == EMegaLightsInput::GBuffer)
		{
			DownsampleMode = FMath::Clamp(CVarMegaLightsDownsampleMode.GetValueOnAnyThread(), 0, 2);
		}
		else if (InputType == EMegaLightsInput::HairStrands)
		{
			DownsampleMode = FMath::Clamp(CVarMegaLightsHairStrandsDownsampleMode.GetValueOnAnyThread(), 0, 2);
		}
		else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
		{
			DownsampleMode = FMath::Clamp(CVarMegaLightsDownsampleMode.GetValueOnAnyThread(), 0, 2);
		}
		else
		{
			checkf(false, TEXT("MegaLight::GetDownsampleFactorXY not implemented"));
			return FIntPoint(0, 0);
		}

		FIntPoint DownsampleFactorXY = FIntPoint(1, 1);
		switch (DownsampleMode)
		{
			case 0: DownsampleFactorXY = FIntPoint(1, 1); break;
			case 1: DownsampleFactorXY = FIntPoint(2, 1); break;
			case 2: DownsampleFactorXY = FIntPoint(2, 2); break;
		}

		const bool bReferenceMode = GetReferenceShadingPassCount(ShaderPlatform) > 1u;
		if (bReferenceMode)
		{
			DownsampleFactorXY = FIntPoint(1, 1);
		}

		return DownsampleFactorXY;
	}

	FIntPoint GetDownsampleFactorXY(StochasticLighting::EMaterialSource MaterialSource, EShaderPlatform ShaderPlatform)
	{
		if (MaterialSource == StochasticLighting::EMaterialSource::GBuffer)
		{
			return GetDownsampleFactorXY(EMegaLightsInput::GBuffer, ShaderPlatform);
		}
		else if (MaterialSource == StochasticLighting::EMaterialSource::HairStrands)
		{
			return GetDownsampleFactorXY(EMegaLightsInput::HairStrands, ShaderPlatform);
		}
		else if (MaterialSource == StochasticLighting::EMaterialSource::FrontLayerGBuffer)
		{
			return GetDownsampleFactorXY(EMegaLightsInput::FrontLayerTranslucency, ShaderPlatform);
		}
		else
		{
			checkf(false, TEXT("MegaLight::GetDownsampleFactorXY not implemented"));
			return FIntPoint(1, 1);
		}
	}

	FIntPoint GetNumSamplesPerPixel2d(int32 NumSamplesPerPixel1d)
	{
		if (NumSamplesPerPixel1d >= 4)
		{
			return FIntPoint(2, 2);
		}
		else if (NumSamplesPerPixel1d >= 2)
		{
			return FIntPoint(2, 1);
		}
		else
		{
			return FIntPoint(1, 1);
		}
	}

	FIntPoint GetNumSamplesPerPixel2d(EMegaLightsInput InputType)
	{
		switch (InputType)
		{
			case EMegaLightsInput::GBuffer: return GetNumSamplesPerPixel2d(CVarMegaLightsNumSamplesPerPixel.GetValueOnAnyThread());
			case EMegaLightsInput::HairStrands: return GetNumSamplesPerPixel2d(CVarMegaLightsHairStrandsNumSamplesPerPixel.GetValueOnAnyThread());
			case EMegaLightsInput::FrontLayerTranslucency: return GetNumSamplesPerPixel2d(CVarMegaLightsNumSamplesPerPixel.GetValueOnAnyThread());
			default: checkf(false, TEXT("MegaLight::GetNumSamplesPerPixel2d not implemented")); return false;
		};
	}

	FIntVector GetNumSamplesPerVoxel3d(int32 NumSamplesPerVoxel1d)
	{
		if (NumSamplesPerVoxel1d >= 4)
		{
			return FIntVector(2, 2, 1);
		}
		else
		{
			return FIntVector(2, 1, 1);
		}
	}

	FMegaLightsViewState::FResources& GetViewState(const FViewInfo& View, EMegaLightsInput InputType)
	{
		static FMegaLightsViewState::FResources DummyState;

		if (View.ViewState)
		{
			switch (InputType)
			{
			case EMegaLightsInput::GBuffer: return View.ViewState->MegaLights.GBuffer;
			case EMegaLightsInput::HairStrands: return View.ViewState->MegaLights.HairStrands;
			case EMegaLightsInput::FrontLayerTranslucency: return View.ViewState->MegaLights.FrontLayerTranslucency;
			default: checkf(false, TEXT("MegaLight::GetViewState not implemented for InputType")); return DummyState;
			};
		}

		return DummyState;
	}

	EMegaLightsDebugMode GetDebugMode()
	{
		return (EMegaLightsDebugMode) FMath::Clamp(CVarMegaLightsDebug.GetValueOnRenderThread(), 0, (uint32)EMegaLightsDebugMode::MAX - 1u);
	}

	bool IsDebugEnabled(EMegaLightsInput InputType)
	{
		EMegaLightsDebugMode DesiredMode = EMegaLightsDebugMode::MAX;
		switch (InputType)
		{
			case EMegaLightsInput::GBuffer: DesiredMode = EMegaLightsDebugMode::GBuffer; break;
			case EMegaLightsInput::HairStrands: DesiredMode = EMegaLightsDebugMode::HairStrands; break;
			case EMegaLightsInput::FrontLayerTranslucency: DesiredMode = EMegaLightsDebugMode::FrontLayerTranslucency; break;
			default: checkf(false, TEXT("MegaLight::IsDebugEnabled not implemented")); return 0;
		};

		return GetDebugMode() == DesiredMode;
	}

	bool IsDebugEnabled(EMegaLightsDebugMode DesiredMode)
	{
		return GetDebugMode() == DesiredMode;
	}

	bool IsDebugEnabledForShadingPass(int32 ShadingPassIndex, EShaderPlatform InPlatform)
	{
		int32 NumPass = GetReferenceShadingPassCount(InPlatform);
		int32 DebuggedPassIndex = CVarMegaLightsReferenceDebuggedPassIndex.GetValueOnRenderThread();
		if (DebuggedPassIndex >= 0)
		{
			return ShadingPassIndex == DebuggedPassIndex;
		}
		else
		{
			return ShadingPassIndex == NumPass + DebuggedPassIndex;
		}
	}

	bool UseFastClear(EMegaLightsInput InputType)
	{
		// Partial clear is a bit slower when entire screen is filled with valid tiles, so disable it for opaque (GBuffer) pass
		switch (InputType)
		{
			case EMegaLightsInput::GBuffer: return CVarMegaLightsFastClear.GetValueOnRenderThread();
			case EMegaLightsInput::HairStrands: return CVarMegaLightsHairStrandsFastClear.GetValueOnRenderThread();
			case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsFrontLayerTranslucencyFastClear.GetValueOnRenderThread();
			default: checkf(false, TEXT("MegaLight::UseFastClear not implemented")); return false;
		};
	}

	bool SupportsSpatialFilter(EMegaLightsInput InputType)
	{
		switch (InputType)
		{
			case EMegaLightsInput::GBuffer: return true;
			case EMegaLightsInput::HairStrands: return false; // Disable for now due to lack of proper reconstruction filter
			case EMegaLightsInput::FrontLayerTranslucency: return true;
			default: checkf(false, TEXT("MegaLight::SupportsSpatialFilter not implemented")); return false;
		};
	}

	bool UseSoftShadows(EMegaLightsInput InputType)
	{
		switch (InputType)
		{
		case EMegaLightsInput::GBuffer: return true;
		case EMegaLightsInput::HairStrands: return true;
		case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsFrontLayerTranslucencySoftShadows.GetValueOnRenderThread();
		default: checkf(false, TEXT("MegaLight::UseSoftShadows not implemented")); return false;
		};
	}

	bool UseTexturedRectLights(EMegaLightsInput InputType)
	{
		switch (InputType)
		{
		case EMegaLightsInput::GBuffer: return true;
		case EMegaLightsInput::HairStrands: return true;
		case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsFrontLayerTranslucencyTexturedRectLights.GetValueOnRenderThread();
		default: checkf(false, TEXT("MegaLight::UseTexturedRectLights not implemented")); return false;
		};
	}

	bool UseWaveOps(EShaderPlatform ShaderPlatform)
	{
		return CVarMegaLightsWaveOps.GetValueOnRenderThread()
			&& GRHISupportsWaveOperations
			&& RHISupportsWaveOperations(ShaderPlatform);
	}

	void ModifyCompilationEnvironment(EShaderPlatform Platform, FShaderCompilerEnvironment& OutEnvironment)
	{
		FForwardLightingParameters::ModifyCompilationEnvironment(Platform, OutEnvironment);
		ShaderPrint::ModifyCompilationEnvironment(Platform, OutEnvironment);

		OutEnvironment.CompilerFlags.Add(CFLAG_WarningsAsErrors);
	}

	const TCHAR* GetTileTypeString(ETileType TileType)
	{
		switch (TileType)
		{
		case ETileType::SimpleShading:						return TEXT("Simple");
		case ETileType::SingleShading:						return TEXT("Single");
		case ETileType::ComplexShading:						return TEXT("Complex");
		case ETileType::ComplexSpecialShading:				return TEXT("Complex Special ");

		case ETileType::SimpleShading_Rect:					return TEXT("Simple Rect");
		case ETileType::SingleShading_Rect:					return TEXT("Single Rect");
		case ETileType::ComplexShading_Rect:				return TEXT("Complex Rect");
		case ETileType::ComplexSpecialShading_Rect:			return TEXT("Complex Special Rect");

		case ETileType::SimpleShading_Rect_Textured:		return TEXT("Simple Textured Rect");
		case ETileType::SingleShading_Rect_Textured:		return TEXT("Single Textured Rect");
		case ETileType::ComplexShading_Rect_Textured:		return TEXT("Complex Textured Rect");
		case ETileType::ComplexSpecialShading_Rect_Textured:return TEXT("Complex Special Textured Rect");

		case ETileType::Empty:								return TEXT("Empty");
		
		default:
			return nullptr;
		}
	}

	bool IsRectLightTileType(ETileType TileType)
	{
		return TileType == MegaLights::ETileType::SimpleShading_Rect
			|| TileType == MegaLights::ETileType::ComplexShading_Rect
			|| TileType == MegaLights::ETileType::SimpleShading_Rect_Textured
			|| TileType == MegaLights::ETileType::ComplexShading_Rect_Textured

			|| TileType == MegaLights::ETileType::SingleShading_Rect
			|| TileType == MegaLights::ETileType::ComplexSpecialShading_Rect
			|| TileType == MegaLights::ETileType::SingleShading_Rect_Textured
			|| TileType == MegaLights::ETileType::ComplexSpecialShading_Rect_Textured;
	}

	bool IsTexturedLightTileType(ETileType TileType)
	{
		return TileType == MegaLights::ETileType::SimpleShading_Rect_Textured
			|| TileType == MegaLights::ETileType::ComplexShading_Rect_Textured
			|| TileType == MegaLights::ETileType::SingleShading_Rect_Textured
			|| TileType == MegaLights::ETileType::ComplexSpecialShading_Rect_Textured;
	}

	bool IsComplexTileType(ETileType TileType)
	{
		return TileType == MegaLights::ETileType::ComplexShading
			|| TileType == MegaLights::ETileType::ComplexSpecialShading
			|| TileType == MegaLights::ETileType::ComplexShading_Rect
			|| TileType == MegaLights::ETileType::ComplexSpecialShading_Rect
			|| TileType == MegaLights::ETileType::ComplexShading_Rect_Textured
			|| TileType == MegaLights::ETileType::ComplexSpecialShading_Rect_Textured;
	}

	int32 GetShadingTileMax(EShaderPlatform InPlatform)
	{
		if (Substrate::IsSubstrateEnabled())
		{
			if (Substrate::IsSubstrateBlendableGBufferEnabled(InPlatform))
			{
				return (int32)MegaLights::ETileType::SHADING_MAX_SUBSTRATE_BLENDABLE;
			}
			else
			{
				return (int32)MegaLights::ETileType::SHADING_MAX_SUBSTRATE_ADAPTIVE;
			}
		}

		return (int32)MegaLights::ETileType::SHADING_MAX_LEGACY;
	}

	TArray<int32> GetShadingTileTypes(EMegaLightsInput InputType, EShaderPlatform InPlatform)
	{
		// Build available tile types
		TArray<int32> Out;
		if (InputType == EMegaLightsInput::GBuffer)
		{
			const int32 ShadingTileMax = GetShadingTileMax(InPlatform);

			for (int32 TileType = 0; TileType < ShadingTileMax; ++TileType)
			{
				if (TileType != (int32)ETileType::Empty)
				{
					Out.Add(TileType);
				}
			}
		}
		else if (InputType == EMegaLightsInput::HairStrands)
		{
			// Hair only uses complex tiles
			Out.Add(int32(MegaLights::ETileType::ComplexShading));
			Out.Add(int32(MegaLights::ETileType::ComplexShading_Rect));
			Out.Add(int32(MegaLights::ETileType::ComplexShading_Rect_Textured));
		}
		else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
		{
			Out.Add(int32(MegaLights::ETileType::SimpleShading));
			Out.Add(int32(MegaLights::ETileType::SimpleShading_Rect));
			Out.Add(int32(MegaLights::ETileType::SimpleShading_Rect_Textured));
		}
		else
		{
			checkf(false, TEXT("MegaLight::GetShadingTileTypes(...) not implemented"))
		}
		return Out;
	}

	void SetupTileClassifyParameters(const FViewInfo& View, MegaLights::FTileClassifyParameters& OutParameters)
	{
		OutParameters.MegaLightsHistoryDistanceThreshold = MegaLights::GetTemporalHistoryDistanceThreshold();
		OutParameters.EnableTexturedRectLights = CVarMegaLightsTexturedRectLights.GetValueOnRenderThread();
	}

	float GetMinSampleWeightEstimate()
	{
		float MinSampleWeightEstimate = 0.0f;

		float LightAttenuationFalloff = CVarMegaLightsLightAttenuationFalloff.GetValueOnRenderThread();
		if (LightAttenuationFalloff > 0.0f)
		{
			MinSampleWeightEstimate = CVarMegaLightsMinSampleWeight.GetValueOnRenderThread() / LightAttenuationFalloff;
		}

		return MinSampleWeightEstimate;
	}
};

namespace MegaLightsVolume
{
	uint32 GetDownsampleFactor(EShaderPlatform ShaderPlatform)
	{
		const uint32 DownsampleMode = FMath::Clamp(CVarMegaLightsVolumeDownsampleMode.GetValueOnAnyThread(), 0, 2);
		uint32 DownsampleFactor = DownsampleMode == 2 ? 2 : 1;

		const bool bReferenceMode = MegaLights::GetReferenceShadingPassCount(ShaderPlatform) > 1u;
		if (bReferenceMode)
		{
			DownsampleFactor = 1;
		}

		return DownsampleFactor;
	}

	FIntVector GetNumSamplesPerVoxel3d()
	{
		return MegaLights::GetNumSamplesPerVoxel3d(CVarMegaLightsVolumeNumSamplesPerVoxel.GetValueOnAnyThread());
	}
}

namespace MegaLightsTranslucencyVolume
{
	uint32 GetDownsampleFactor(EShaderPlatform ShaderPlatform, bool bUnifiedVolume)
	{
		if (bUnifiedVolume)
		{
			return MegaLightsVolume::GetDownsampleFactor(ShaderPlatform);
		}

		uint32 DownsampleFactor = FMath::Clamp(CVarMegaLightsTranslucencyVolumeDownsampleFactor.GetValueOnAnyThread(), 1, 2);
		
		const bool bReferenceMode = MegaLights::GetReferenceShadingPassCount(ShaderPlatform) > 1u;
		if (bReferenceMode)
		{
			DownsampleFactor = 1;
		}

		return DownsampleFactor;
	}

	FIntVector GetNumSamplesPerVoxel3d(bool bUnifiedVolume)
	{
		if (bUnifiedVolume)
		{
			return MegaLightsVolume::GetNumSamplesPerVoxel3d();
		}

		return MegaLights::GetNumSamplesPerVoxel3d(CVarMegaLightsTranslucencyVolumeNumSamplesPerVoxel.GetValueOnAnyThread());
	}
}

class FMegaLightsTileClassificationBuildListsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsTileClassificationBuildListsCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsTileClassificationBuildListsCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWTileAllocator)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWTileData)
		SHADER_PARAMETER(FIntPoint, ViewSizeInTiles)
		SHADER_PARAMETER(FIntPoint, ViewMinInTiles)
		SHADER_PARAMETER(FIntPoint, DownsampledViewSizeInTiles)
		SHADER_PARAMETER(FIntPoint, DownsampledViewMinInTiles)
		SHADER_PARAMETER(uint32, OutputTileDataStride)
		SHADER_PARAMETER(uint32, UseTexturedRectLights)
	END_SHADER_PARAMETER_STRUCT()

	class FDownsampleFactorX : SHADER_PERMUTATION_RANGE_INT("DOWNSAMPLE_FACTOR_X", 1, 2);
	class FDownsampleFactorY : SHADER_PERMUTATION_RANGE_INT("DOWNSAMPLE_FACTOR_Y", 1, 2);
	class FFastClear : SHADER_PERMUTATION_BOOL("FAST_CLEAR");
	using FPermutationDomain = TShaderPermutationDomain<FDownsampleFactorX, FDownsampleFactorY, FFastClear>;

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FDownsampleFactorY>() == 2)
		{
			PermutationVector.Set<FDownsampleFactorX>(2);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
		OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
	}

	static int32 GetGroupSize()
	{
		return 16;
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsTileClassificationBuildListsCS, "/Engine/Private/MegaLights/MegaLights.usf", "MegaLightsTileClassificationBuildListsCS", SF_Compute);

class FInitTileIndirectArgsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FInitTileIndirectArgsCS)
	SHADER_USE_PARAMETER_STRUCT(FInitTileIndirectArgsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWTileIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWDownsampledTileIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileAllocator)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, DownsampledTileAllocator)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}

	static int32 GetGroupSize()
	{
		return 64;
	}
};

IMPLEMENT_GLOBAL_SHADER(FInitTileIndirectArgsCS, "/Engine/Private/MegaLights/MegaLights.usf", "InitTileIndirectArgsCS", SF_Compute);

DECLARE_GPU_STAT(MegaLights);
DECLARE_GPU_STAT(MegaLightsFrontLayerTranslucency);

extern int32 GetTranslucencyLightingVolumeDim();
extern float GetTranslucencyLightingVolumeOuterDistance();

static int32 GetVolumeGridPixelSize()
{
	return FMath::Max(1, CVarMegaLightsVolumeGridPixelSize.GetValueOnRenderThread());
}

static int32 GetVolumeGridSizeZ()
{
	return FMath::Max(1, CVarMegaLightsVolumeGridSizeZ.GetValueOnRenderThread());
}

static FVector GetVolumeGridZParams(float VolumeStartDistance, float NearPlane, float FarPlane, int32 GridSizeZ)
{
	// Don't spend lots of resolution right in front of the near plane
	NearPlane = FMath::Max(NearPlane, VolumeStartDistance);

	return CalculateGridZParams(NearPlane, FarPlane, CVarMegaLightsVolumeDepthDistributionScale.GetValueOnRenderThread(), GridSizeZ);
}

static FIntVector GetVolumeGridSize(const FIntPoint& TargetResolution, int32& OutGridPixelSize)
{
	OutGridPixelSize = GetVolumeGridPixelSize();
	return GetFroxelGridSize(TargetResolution, OutGridPixelSize, GetVolumeGridSizeZ());
}

FIntVector GetVolumeResourceGridSize(const FViewInfo& View, int32& OutGridPixelSize)
{
	return GetVolumeGridSize(View.GetSceneTexturesConfig().Extent, OutGridPixelSize);
}

FIntVector GetVolumeViewGridSize(const FViewInfo& View, int32& OutGridPixelSize)
{
	return GetVolumeGridSize(View.ViewRect.Size(), OutGridPixelSize);
}

void SetupMegaLightsVolumeData(const FViewInfo& View, bool bShouldRenderVolumetricFog, bool bShouldRenderTranslucencyVolume, FMegaLightsVolumeData& Parameters)
{
	const FScene* Scene = (FScene*)View.Family->Scene;

	float MaxDistance = 0.0f;

	{
		if (bShouldRenderVolumetricFog && Scene->ExponentialFogs.Num() > 0)
		{
			const FExponentialHeightFogSceneInfo& FogInfo = Scene->ExponentialFogs[0];

			MaxDistance = FMath::Max(MaxDistance, FogInfo.VolumetricFogDistance);
		}
		
		if (bShouldRenderTranslucencyVolume)
		{
			MaxDistance = FMath::Max(MaxDistance, GetTranslucencyLightingVolumeOuterDistance());
		}
	}

	int32 VolumeGridPixelSize;
	const FIntVector VolumeViewGridSize = GetVolumeViewGridSize(View, VolumeGridPixelSize);
	const FIntVector VolumeResourceGridSize = GetVolumeResourceGridSize(View, VolumeGridPixelSize);

	Parameters.ViewGridSizeInt = VolumeViewGridSize;
	Parameters.ViewGridSize = FVector3f(VolumeViewGridSize);
	Parameters.ResourceGridSizeInt = VolumeResourceGridSize;
	Parameters.ResourceGridSize = FVector3f(VolumeResourceGridSize);

	FVector ZParams = GetVolumeGridZParams(0, View.NearClippingDistance, MaxDistance, VolumeResourceGridSize.Z);
	Parameters.GridZParams = (FVector3f)ZParams;

	Parameters.SVPosToVolumeUV = FVector2f::UnitVector / (FVector2f(VolumeResourceGridSize.X, VolumeResourceGridSize.Y) * VolumeGridPixelSize);
	Parameters.FogGridToPixelXY = FIntPoint(VolumeGridPixelSize, VolumeGridPixelSize);
	Parameters.MaxDistance = MaxDistance;
}

FMegaLightsViewContext::FMegaLightsViewContext(
	FRDGBuilder& InGraphBuilder,
	const int32 InViewIndex,
	const FViewInfo& InView,
	const FSceneViewFamily& InViewFamily,
	const FScene* InScene,
	const FSceneTextures& InSceneTextures,
	bool bInUseVSM)
	: GraphBuilder(InGraphBuilder)
	, ViewIndex(InViewIndex)
	, View(InView)
	, ViewFamily(InViewFamily)
	, Scene(InScene)
	, SceneTextures(InSceneTextures)
	, bUseVSM(bInUseVSM)
{
	StochasticLighting::InitDefaultHistoryScreenParameters(HistoryScreenParameters);
}

void FMegaLightsViewContext::TileClassificationMark(uint32 ShadingPassIndex, ERDGPassFlags ComputePassFlags)
{
	if (ShadingPassIndex > 0 && DownsampleFactor == FIntPoint(1, 1))
	{
		// only need to rerun tile classification pass for the different shading passes when downsampling is enabled
		check(MegaLightsParameters.MegaLightsTileBitmask != nullptr);
		check(HistoryScreenParameters.EncodedHistoryScreenCoordTexture != nullptr);
		check(HistoryScreenParameters.PackedPixelDataTexture != nullptr);
		return;
	}

	const FIntPoint BufferSizeInTiles = FMath::DivideAndRoundUp<FIntPoint>(SceneTextures.Config.Extent, MegaLights::TileSize);

	const FFrontLayerTranslucencyGBufferParameters FrontLayerTranslucencyGBufferParameters = GetFrontLayerTranslucencyGBufferParameters(FrontLayerTranslucencyData ? *FrontLayerTranslucencyData : FFrontLayerTranslucencyData());

	StochasticLighting::EMaterialSource MaterialSource = StochasticLighting::EMaterialSource::MAX;
	switch (InputType)
	{
	case EMegaLightsInput::GBuffer:
		MaterialSource = StochasticLighting::EMaterialSource::GBuffer;
		break;
	case EMegaLightsInput::HairStrands:
		MaterialSource = StochasticLighting::EMaterialSource::HairStrands;
		break;
	case EMegaLightsInput::FrontLayerTranslucency:
		MaterialSource = StochasticLighting::EMaterialSource::FrontLayerGBuffer;
		break;
	default:
		checkNoEntry();
		return;
	}

	StochasticLighting::FRunConfig RunConfig(View);
	RunConfig.ComputePassFlags = ComputePassFlags;
	RunConfig.ReflectionsMethod = EReflectionsMethod::Disabled;
	RunConfig.StateFrameIndexOverride = ShadingPassIndex == 0 ? -1 : FirstPassStateFrameIndex + ShadingPassIndex;
	RunConfig.bCopyDepthAndNormal = (ShadingPassIndex == 0);
	RunConfig.bDownsampleDepthAndNormal2x1 = (DownsampleFactor == FIntPoint(2, 1));
	RunConfig.bDownsampleDepthAndNormal2x2 = (DownsampleFactor == FIntPoint(2, 2));
	RunConfig.bTileClassifyMegaLights = true;
	RunConfig.bReprojectMegaLights = true;
	// MegaLights downsamples depth/normal in GenerateLightSamplesCS as it is faster
	RunConfig.bOutputDownsampledDepthAndNormal = false;

	StochasticLighting::FFrameTemporaries StochasticLightingFrameTemporaries = StochasticLighting::TileClassificationMark(
		GraphBuilder,
		MakeConstArrayView(&View, 1),
		SceneTextures,
		FrontLayerTranslucencyGBufferParameters,
		MakeConstArrayView(&RunConfig, 1),
		MaterialSource);

	if (RunConfig.bCopyDepthAndNormal)
	{
		// DepthHistory / NormalHistory are copies of current frame scene depth / normal and will be used as history next frame
		SceneDepth = StochasticLightingFrameTemporaries.DepthHistory.GetRenderTarget();
		SceneWorldNormal = StochasticLightingFrameTemporaries.NormalHistory.GetRenderTarget();
	}

	if (MegaLightsParameters.MegaLightsTileBitmask == nullptr)
	{
		checkf(ShadingPassIndex == 0, TEXT("MegaLightsTileBitmask should've been generated by shading pass 0."));
		MegaLightsParameters.MegaLightsTileBitmask = StochasticLightingFrameTemporaries.MegaLightsTileBitmask.GetRenderTarget();
	}
	HistoryScreenParameters.EncodedHistoryScreenCoordTexture = StochasticLightingFrameTemporaries.EncodedHistoryScreenCoord.GetRenderTarget();
	HistoryScreenParameters.PackedPixelDataTexture = StochasticLightingFrameTemporaries.MegaLightsPackedPixelData.GetRenderTarget();
}

void FMegaLightsViewContext::SetupStageOne(
	const bool bInShouldRenderVolumetricFog,
	const bool bInShouldRenderTranslucencyVolume,
	TUniformBufferRef<FBlueNoise> BlueNoiseUniformBuffer, 
	EMegaLightsInput InInputType)
{
	// History reset for debugging purposes
	bool bResetHistory = false;

	if (GMegaLightsResetEveryNthFrame > 0 && (ViewFamily.FrameNumber % (uint32)GMegaLightsResetEveryNthFrame) == 0)
	{
		bResetHistory = true;
	}

	if (GMegaLightsReset != 0)
	{
		GMegaLightsReset = 0;
		bResetHistory = true;
	}

	InputType = InInputType;

	bShouldRenderVolumetricFog = bInShouldRenderVolumetricFog;
	bShouldRenderTranslucencyVolume = bInShouldRenderTranslucencyVolume;

	bUnifiedVolume = MegaLights::UseVolume() && CVarMegaLightsVolumeUnified.GetValueOnRenderThread() != 0;
	bVolumeEnabled = MegaLights::UseVolume() && (bShouldRenderVolumetricFog || (bUnifiedVolume && bShouldRenderTranslucencyVolume));

	bDebug = MegaLights::IsDebugEnabled(InInputType);
	bVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::Volume);
	bTranslucencyVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::TranslucencyVolume);
	DebugTileClassificationMode = MegaLights::GetDebugTileClassificationMode();
	VisualizeLightComplexityTarget = MegaLights::GetVisualizeLightComplexityTarget(View);
	bVisualizeLightComplexityFrozen = MegaLights::IsVisualizeLightComplexityFrozen();
	bVisualizeLightComplexityDump = GMegaLightsVisualizeLightComplexityDump != 0;

	if (InputType == EMegaLightsInput::GBuffer)
	{
		VisualizeLightComplexityTarget &= EMegaLightsDebugTarget::GBuffer | EMegaLightsDebugTarget::VolumeMask;
	}
	else if (InputType == EMegaLightsInput::HairStrands)
	{
		VisualizeLightComplexityTarget &= EMegaLightsDebugTarget::HairStrands;
	}
	else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
	{
		VisualizeLightComplexityTarget &= EMegaLightsDebugTarget::FrontLayerTranslucency;
	}
	else
	{
		unimplemented();
	}

	NumSamplesPerPixel2d = MegaLights::GetNumSamplesPerPixel2d(InputType);
	NumSamplesPerVoxel3d = MegaLightsVolume::GetNumSamplesPerVoxel3d();
	NumSamplesPerTranslucencyVoxel3d = MegaLightsTranslucencyVolume::GetNumSamplesPerVoxel3d(bUnifiedVolume);

	DownsampleFactor = MegaLights::GetDownsampleFactorXY(InputType, View.GetShaderPlatform());
		const FIntPoint DownsampledViewSize = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), DownsampleFactor);
		const FIntPoint SampleViewSize = DownsampledViewSize * NumSamplesPerPixel2d;
		const FIntPoint DownsampledBufferSize = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, DownsampleFactor);
	SampleBufferSize = DownsampledBufferSize * NumSamplesPerPixel2d;
	DonwnsampledSampleBufferSize = DownsampledBufferSize * NumSamplesPerPixel2d;

	DownsampledSceneDepth = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_R32_FLOAT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("DownsampledSceneDepth"));

	DownsampledSceneWorldNormal = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_A2B10G10R10, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("DownsampledSceneWorldNormal"));

	LightSamples = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(DonwnsampledSampleBufferSize, PF_R32_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("LightSamples"));

	LightSampleRays = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(DonwnsampledSampleBufferSize, PF_R32_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("LightSampleRays"));

	bSpatial = MegaLights::UseSpatialFilter(InputType);
	bTemporal = MegaLights::UseTemporalFilter(InputType);

	VisibleLightHashSizeInTiles = FMath::DivideAndRoundUp<FIntPoint>(SceneTextures.Config.Extent, MegaLights::VisibleLightHashTileSize);
	VisibleLightHashViewMinInTiles = FMath::DivideAndRoundUp<FIntPoint>(View.ViewRect.Min, MegaLights::VisibleLightHashTileSize);
	VisibleLightHashViewSizeInTiles = FMath::DivideAndRoundUp<FIntPoint>(View.ViewRect.Size(), MegaLights::VisibleLightHashTileSize);

	VisibleLightHashBufferSize = VisibleLightHashSizeInTiles.X * VisibleLightHashSizeInTiles.Y * MegaLights::VisibleLightHashSize;
	HiddenLightHashBufferSize = VisibleLightHashSizeInTiles.X * VisibleLightHashSizeInTiles.Y * MegaLights::HiddenLightHashSize;

	SetupMegaLightsVolumeData(View, bShouldRenderVolumetricFog, bShouldRenderTranslucencyVolume, VolumeParameters);

	if (bShouldRenderVolumetricFog)
	{
		SetupVolumetricFogGlobalData(View, VolumetricFogParamaters);
	}

	if (!bUnifiedVolume)
	{
		VolumeParameters.ViewGridSizeInt = VolumetricFogParamaters.ViewGridSizeInt;
		VolumeParameters.ViewGridSize = VolumetricFogParamaters.ViewGridSize;
		VolumeParameters.ResourceGridSizeInt = VolumetricFogParamaters.ResourceGridSizeInt;
		VolumeParameters.ResourceGridSize = VolumetricFogParamaters.ResourceGridSize;
		VolumeParameters.GridZParams = VolumetricFogParamaters.GridZParams;
		VolumeParameters.SVPosToVolumeUV = VolumetricFogParamaters.SVPosToVolumeUV;
		VolumeParameters.FogGridToPixelXY = VolumetricFogParamaters.FogGridToPixelXY;
		VolumeParameters.MaxDistance = VolumetricFogParamaters.MaxDistance;
	}

	VolumeDownsampleFactor = MegaLightsVolume::GetDownsampleFactor(View.GetShaderPlatform());
	VolumeViewSize = VolumeParameters.ViewGridSizeInt;
	VolumeBufferSize = VolumeParameters.ResourceGridSizeInt;
		const FIntVector VolumeDownsampledBufferSize = FIntVector::DivideAndRoundUp(VolumeParameters.ResourceGridSizeInt, VolumeDownsampleFactor);
	VolumeDownsampledViewSize = FIntVector::DivideAndRoundUp(VolumeParameters.ViewGridSizeInt, VolumeDownsampleFactor);
		const FIntVector VolumeSampleViewSize = VolumeDownsampledViewSize * NumSamplesPerVoxel3d;
	VolumeSampleBufferSize = VolumeDownsampledBufferSize * NumSamplesPerVoxel3d;

	VolumeVisibleLightHashTileSize = FIntVector(4, 4, 2);

	VolumeVisibleLightHashSizeInTiles = FIntVector(
			FMath::DivideAndRoundUp(VolumeBufferSize.X, VolumeVisibleLightHashTileSize.X),
			FMath::DivideAndRoundUp(VolumeBufferSize.Y, VolumeVisibleLightHashTileSize.Y),
			FMath::DivideAndRoundUp(VolumeBufferSize.Z, VolumeVisibleLightHashTileSize.Z));

	VolumeVisibleLightHashViewSizeInTiles = FIntVector(
			FMath::DivideAndRoundUp(VolumeViewSize.X, VolumeVisibleLightHashTileSize.X),
			FMath::DivideAndRoundUp(VolumeViewSize.Y, VolumeVisibleLightHashTileSize.Y),
			FMath::DivideAndRoundUp(VolumeViewSize.Z, VolumeVisibleLightHashTileSize.Z));

	VolumeVisibleLightHashBufferSize = VolumeVisibleLightHashSizeInTiles.X * VolumeVisibleLightHashSizeInTiles.Y * VolumeVisibleLightHashSizeInTiles.Z * MegaLights::VisibleLightHashSize;
	VolumeHiddenLightHashBufferSize = VolumeVisibleLightHashSizeInTiles.X * VolumeVisibleLightHashSizeInTiles.Y * VolumeVisibleLightHashSizeInTiles.Z * MegaLights::HiddenLightHashSize;

	TranslucencyVolumeDownsampleFactor = MegaLightsTranslucencyVolume::GetDownsampleFactor(View.GetShaderPlatform(), bUnifiedVolume);
	TranslucencyVolumeBufferSize = FIntVector(GetTranslucencyLightingVolumeDim());
	TranslucencyVolumeDownsampledBufferSize = bUnifiedVolume ? VolumeDownsampledBufferSize : FIntVector::DivideAndRoundUp(TranslucencyVolumeBufferSize, TranslucencyVolumeDownsampleFactor);
		const FIntVector TranslucencyVolumeDownsampledViewSize = bUnifiedVolume ? VolumeDownsampledViewSize : TranslucencyVolumeDownsampledBufferSize;
	TranslucencyVolumeSampleBufferSize = bUnifiedVolume ? VolumeSampleBufferSize : TranslucencyVolumeDownsampledBufferSize * NumSamplesPerTranslucencyVoxel3d;

	TranslucencyVolumeVisibleLightHashTileSize = FIntVector(2, 2, 2);

	TranslucencyVolumeVisibleLightHashSizeInTiles = FIntVector(
			FMath::DivideAndRoundUp(TranslucencyVolumeBufferSize.X, TranslucencyVolumeVisibleLightHashTileSize.X),
			FMath::DivideAndRoundUp(TranslucencyVolumeBufferSize.Y, TranslucencyVolumeVisibleLightHashTileSize.Y),
			FMath::DivideAndRoundUp(TranslucencyVolumeBufferSize.Z, TranslucencyVolumeVisibleLightHashTileSize.Z));

	TranslucencyVolumeVisibleLightHashBufferSize =
			TranslucencyVolumeVisibleLightHashSizeInTiles.X *
			TranslucencyVolumeVisibleLightHashSizeInTiles.Y *
			TranslucencyVolumeVisibleLightHashSizeInTiles.Z *
			MegaLights::VisibleLightHashSize;

	TranslucencyVolumeHiddenLightHashBufferSize =
		TranslucencyVolumeVisibleLightHashSizeInTiles.X *
		TranslucencyVolumeVisibleLightHashSizeInTiles.Y *
		TranslucencyVolumeVisibleLightHashSizeInTiles.Z *
		MegaLights::HiddenLightHashSize;

	bGuideByHistory = CVarMegaLightsGuideByHistory.GetValueOnRenderThread();
	bVolumeGuideByHistory = CVarMegaLightsVolumeGuideByHistory.GetValueOnRenderThread() != 0;
	bTranslucencyVolumeGuideByHistory = CVarMegaLightsTranslucencyVolumeGuideByHistory.GetValueOnRenderThread() != 0;
	bSubPixelShading = CVarMegaLightsHairStrandsSubPixelShading.GetValueOnRenderThread() > 0;
	bUseHairComplexTransmittance = InputType == EMegaLightsInput::HairStrands
		|| (View.HairCardsMeshElements.Num() && IsHairStrandsSupported(EHairStrandsShaderType::All, View.GetShaderPlatform()))
		|| (IsHairComplexTransmittanceEnabled(View.GetShaderPlatform()) && IsHairStrandsSupported(EHairStrandsShaderType::All, View.GetShaderPlatform()));

	if (View.ViewState)
	{
		const FMegaLightsViewState::FResources& MegaLightsViewState = MegaLights::GetViewState(View, InputType);
		const FStochasticLightingViewState& StochasticLightingViewState = View.ViewState->StochasticLighting;

		if (!View.bCameraCut 
			&& !View.bPrevTransformsReset
			&& !bResetHistory)
		{
			HistoryVisibleLightHashViewMinInTiles = MegaLightsViewState.HistoryVisibleLightHashViewMinInTiles;
			HistoryVisibleLightHashViewSizeInTiles = MegaLightsViewState.HistoryVisibleLightHashViewSizeInTiles;

			HistoryVolumeVisibleLightHashViewSizeInTiles = MegaLightsViewState.HistoryVolumeVisibleLightHashViewSizeInTiles;
			HistoryTranslucencyVolumeVisibleLightHashSizeInTiles = MegaLightsViewState.HistoryTranslucencyVolumeVisibleLightHashSizeInTiles;

			if (InputType == EMegaLightsInput::GBuffer)
			{
				// GBuffer uses StochasticLightingViewState
				SceneDepthHistory = TryRegisterExternalTexture(GraphBuilder, StochasticLightingViewState.SceneDepthHistory);
				SceneNormalAndShadingHistory = TryRegisterExternalTexture(GraphBuilder, StochasticLightingViewState.SceneNormalHistory);
			}
			else if (InputType == EMegaLightsInput::HairStrands)
			{
				SceneDepthHistory = TryRegisterExternalTexture(GraphBuilder, MegaLightsViewState.SceneDepthHistory);
				SceneNormalAndShadingHistory = TryRegisterExternalTexture(GraphBuilder, MegaLightsViewState.SceneNormalHistory);
			}
			else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
			{
				// FrontLayerTranslucency uses StochasticLightingViewState
				SceneDepthHistory = TryRegisterExternalTexture(GraphBuilder, StochasticLightingViewState.FrontLayerTranslucencyDepthHistory);
				SceneNormalAndShadingHistory = TryRegisterExternalTexture(GraphBuilder, StochasticLightingViewState.FrontLayerTranslucencyNormalHistory);
			}
			else
			{
				unimplemented();
			}

			if (bTemporal &&
				MegaLightsViewState.DiffuseLightingHistory
				&& MegaLightsViewState.SpecularLightingHistory
				&& MegaLightsViewState.LightingMomentsHistory
				&& MegaLightsViewState.NumFramesAccumulatedHistory)
			{
				DiffuseLightingHistory = GraphBuilder.RegisterExternalTexture(MegaLightsViewState.DiffuseLightingHistory);
				SpecularLightingHistory = GraphBuilder.RegisterExternalTexture(MegaLightsViewState.SpecularLightingHistory);
				LightingMomentsHistory = GraphBuilder.RegisterExternalTexture(MegaLightsViewState.LightingMomentsHistory);
				NumFramesAccumulatedHistory = GraphBuilder.RegisterExternalTexture(MegaLightsViewState.NumFramesAccumulatedHistory);
			}

			if (bGuideByHistory
				&& MegaLightsViewState.VisibleLightHashHistory
				&& MegaLightsViewState.HiddenLightHashHistory)
			{
				VisibleLightHashHistory = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.VisibleLightHashHistory);
				HiddenLightHashHistory = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.HiddenLightHashHistory);
			}

			if (bVolumeGuideByHistory
				&& MegaLightsViewState.VolumeVisibleLightHashHistory
				&& MegaLightsViewState.VolumeHiddenLightHashHistory)
			{
				VolumeVisibleLightHashHistory = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.VolumeVisibleLightHashHistory);
				VolumeHiddenLightHashHistory = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.VolumeHiddenLightHashHistory);
			}

			if (bTranslucencyVolumeGuideByHistory
				&& MegaLightsViewState.TranslucencyVolume0VisibleLightHashHistory
				&& MegaLightsViewState.TranslucencyVolume1VisibleLightHashHistory
				&& MegaLightsViewState.TranslucencyVolume0HiddenLightHashHistory
				&& MegaLightsViewState.TranslucencyVolume1HiddenLightHashHistory
				&& TranslucencyVolumeVisibleLightHashBufferSize == MegaLightsViewState.TranslucencyVolume0VisibleLightHashHistory->GetSize() / sizeof(uint32)
				&& TranslucencyVolumeVisibleLightHashBufferSize == MegaLightsViewState.TranslucencyVolume1VisibleLightHashHistory->GetSize() / sizeof(uint32))
			{
				TranslucencyVolumeVisibleLightHashHistory[0] = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.TranslucencyVolume0VisibleLightHashHistory);
				TranslucencyVolumeVisibleLightHashHistory[1] = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.TranslucencyVolume1VisibleLightHashHistory);
				TranslucencyVolumeHiddenLightHashHistory[0] = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.TranslucencyVolume0HiddenLightHashHistory);
				TranslucencyVolumeHiddenLightHashHistory[1] = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState.TranslucencyVolume1HiddenLightHashHistory);
			}
		}

		if (MegaLightsViewState.LastVisualizeLightComplexityTarget != VisualizeLightComplexityTarget)
		{
			// Disable the following so we clear the old frozen data which is no longer relevant
			bVisualizeLightComplexityFrozen = false;
			bVisualizeLightComplexityDump = false;
		}
	}

	// Setup the light function atlas
	bUseLightFunctionAtlas = LightFunctionAtlas::IsEnabled(View, LightFunctionAtlas::ELightFunctionAtlasSystem::MegaLights);

	ViewSizeInTiles = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), MegaLights::TileSize);
	const int32 TileDataStride = ViewSizeInTiles.X * ViewSizeInTiles.Y;

	DownsampledViewSizeInTiles = FIntPoint::DivideAndRoundUp(DownsampledViewSize, MegaLights::TileSize);
	const int32 DownsampledTileDataStride = DownsampledViewSizeInTiles.X * DownsampledViewSizeInTiles.Y;

	const FFrontLayerTranslucencyGBufferParameters FrontLayerTranslucencyGBufferParameters = GetFrontLayerTranslucencyGBufferParameters(FrontLayerTranslucencyData ? *FrontLayerTranslucencyData : FFrontLayerTranslucencyData());

	{
		// Defaults to -2 to avoid selecting simple lights whose LightIds are -1
		const int32 InvalidDebugLightId = INDEX_NONE - 1;

		MegaLightsParameters.ViewUniformBuffer = View.ViewUniformBuffer;
		MegaLightsParameters.Scene = View.GetSceneUniforms().GetBuffer(GraphBuilder);
		MegaLightsParameters.SceneTextures = GetSceneTextureParameters(GraphBuilder, SceneTextures.UniformBuffer);
		MegaLightsParameters.SceneTexturesStruct = CreateSceneTextureUniformBuffer(
			GraphBuilder,
			&SceneTextures,
			View.GetFeatureLevel(),
			ESceneTextureSetupMode::SceneDepth | ESceneTextureSetupMode::GBuffers);
		MegaLightsParameters.Substrate = Substrate::BindSubstrateGlobalUniformParameters(View);
		MegaLightsParameters.HairStrands = HairStrands::BindHairStrandsViewUniformParameters(View);
		MegaLightsParameters.ForwardLightStruct = View.ForwardLightingResources.ForwardLightUniformBuffer;
		MegaLightsParameters.LightFunctionAtlas = LightFunctionAtlas::BindGlobalParameters(GraphBuilder, View);
		MegaLightsParameters.LightingChannelParameters = GetSceneLightingChannelParameters(GraphBuilder, View, nullptr);
		MegaLightsParameters.FrontLayerTranslucencyGBufferParameters = FrontLayerTranslucencyGBufferParameters;
		MegaLightsParameters.BlueNoise = BlueNoiseUniformBuffer;
		MegaLightsParameters.PreIntegratedGF = GSystemTextures.PreintegratedGF->GetRHI();
		MegaLightsParameters.PreIntegratedGFSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		MegaLightsParameters.UnjitteredClipToTranslatedWorld = FMatrix44f(View.ViewMatrices.ComputeInvProjectionNoAAMatrix() * View.ViewMatrices.GetTranslatedViewMatrix().GetTransposed()); // LWC_TODO: Precision loss?
		MegaLightsParameters.UnjitteredTranslatedWorldToClip = FMatrix44f(View.ViewMatrices.GetTranslatedViewMatrix() * View.ViewMatrices.ComputeProjectionNoAAMatrix());
		MegaLightsParameters.UnjitteredPrevTranslatedWorldToClip = FMatrix44f(FTranslationMatrix(-View.ViewMatrices.GetPreViewTranslation()) * View.PrevViewInfo.ViewMatrices.GetWorldToView() * View.PrevViewInfo.ViewMatrices.ComputeProjectionNoAAMatrix());

		MegaLightsParameters.DownsampledViewMin = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, DownsampleFactor);
		MegaLightsParameters.DownsampledViewSize = DownsampledViewSize;
		MegaLightsParameters.SampleViewMin = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, DownsampleFactor) * NumSamplesPerPixel2d;
		MegaLightsParameters.SampleViewSize = SampleViewSize;
		MegaLightsParameters.DownsampleFactor = DownsampleFactor;
		MegaLightsParameters.NumSamplesPerPixel = NumSamplesPerPixel2d;
		MegaLightsParameters.NumSamplesPerPixelDivideShift.X = FMath::FloorLog2(NumSamplesPerPixel2d.X);
		MegaLightsParameters.NumSamplesPerPixelDivideShift.Y = FMath::FloorLog2(NumSamplesPerPixel2d.Y);
		MegaLightsParameters.MegaLightsStateFrameIndex = MegaLights::GetStateFrameIndex(View.ViewState, View.GetShaderPlatform());
		MegaLightsParameters.StochasticLightingStateFrameIndex = StochasticLighting::GetStateFrameIndex(View.ViewState);
		MegaLightsParameters.DownsampledSceneDepth = DownsampledSceneDepth;
		MegaLightsParameters.DownsampledSceneWorldNormal = DownsampledSceneWorldNormal;
		MegaLightsParameters.DownsampledBufferInvSize = FVector2f(1.0f) / DownsampledBufferSize;
		MegaLightsParameters.MinSampleWeight = FMath::Max(CVarMegaLightsMinSampleWeight.GetValueOnRenderThread(), 0.0f);
		MegaLightsParameters.MinSampleWeightEstimate = MegaLights::GetMinSampleWeightEstimate();
		MegaLightsParameters.TileDataStride = TileDataStride;
		MegaLightsParameters.DownsampledTileDataStride = DownsampledTileDataStride;
		MegaLightsParameters.DebugCursorPosition.X = CVarMegaLightsDebugCursorX.GetValueOnRenderThread();
		MegaLightsParameters.DebugCursorPosition.Y = CVarMegaLightsDebugCursorY.GetValueOnRenderThread();
		MegaLightsParameters.DebugLightId = InvalidDebugLightId;
		MegaLightsParameters.DebugVisualizeLight = CVarMegaLightsDebugVisualizeLight.GetValueOnRenderThread();
		MegaLightsParameters.DebugVisualizeTraces = CVarMegaLightsDebugVisualizeTraces.GetValueOnRenderThread();
		MegaLightsParameters.UseIESProfiles = CVarMegaLightsIESProfiles.GetValueOnRenderThread() != 0;
		MegaLightsParameters.UseLightFunctionAtlas = bUseLightFunctionAtlas;
		MegaLightsParameters.FrontLayerTranslucencySpecularOnly = CVarMegaLightsFrontLayerTranslucencySpecularOnly.GetValueOnRenderThread() ? 1 : 0;
		
		if (View.ForwardLightingResources.SelectedForwardDirectionalLightProxy
			&& View.ForwardLightingResources.SelectedForwardDirectionalLightIndex >= 0
			&& View.ForwardLightingResources.ForwardLightUniformParameters
			&& View.ForwardLightingResources.ForwardLightUniformParameters->DirectionalLightHandledByMegaLights != 0)
		{
			const FLightSceneInfo* LightSceneInfo = View.ForwardLightingResources.SelectedForwardDirectionalLightProxy->GetLightSceneInfo();
			MegaLightsParameters.bCloudShadowEnabled = SetupLightCloudTransmittanceParameters(GraphBuilder, Scene, View, LightSceneInfo, MegaLightsParameters.CloudShadow);
			MegaLightsParameters.SelectedForwardDirectionalLightIndex = View.ForwardLightingResources.SelectedForwardDirectionalLightIndex;
			MegaLightsParameters.bDirectionalLightPerPixelTransmittanceEnabled = View.ForwardLightingResources.SelectedForwardDirectionalLightProxy->GetPerPixelTransmittanceEnabled() ? 1 : 0;
		}
		else
		{
			MegaLightsParameters.bCloudShadowEnabled = SetupLightCloudTransmittanceParameters(GraphBuilder, nullptr, View, nullptr, MegaLightsParameters.CloudShadow);
			MegaLightsParameters.SelectedForwardDirectionalLightIndex = INDEX_NONE;
			MegaLightsParameters.bDirectionalLightPerPixelTransmittanceEnabled = 0;
		}

		// If editor is disabled then we don't have a valid cursor position and have to force it to the center of the screen
		if (!GIsEditor && (MegaLightsParameters.DebugCursorPosition.X < 0 || MegaLightsParameters.DebugCursorPosition.Y < 0))
		{		
			MegaLightsParameters.DebugCursorPosition.X = View.ViewRect.Min.X + View.ViewRect.Width() / 2;
			MegaLightsParameters.DebugCursorPosition.Y = View.ViewRect.Min.Y + View.ViewRect.Height() / 2;
		}

		// screen traces use ClosestHZB, volume sampling/shading uses FurthestHZB
		MegaLightsParameters.HZBParameters = GetHZBParameters(GraphBuilder, View, EHZBType::All);
		MegaLightsParameters.VisibleLightHashViewMinInTiles = VisibleLightHashViewMinInTiles;
		MegaLightsParameters.VisibleLightHashViewSizeInTiles = VisibleLightHashViewSizeInTiles;

		if (bDebug 
			|| bVolumeDebug 
			|| bTranslucencyVolumeDebug 
			|| DebugTileClassificationMode != 0 
			|| VisualizeLightComplexityTarget != EMegaLightsDebugTarget::None
			|| MegaLights::IsVisualizeRaysEnabled()
			|| MegaLights::ShouldAddVisualizePostProcessingPass(View))
		{
			const FIntPoint TileCountXY = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), MegaLights::TileSize);
			const uint32 TileCount = TileCountXY.X * TileCountXY.Y;
			const uint32 SampleCount = SampleViewSize.X * SampleViewSize.Y;

			ShaderPrint::SetEnabled(true);
			ShaderPrint::RequestSpaceForLines(1024u + FMath::Max(TileCount * 4u, SampleCount));
			ShaderPrint::RequestSpaceForTriangles(1024u + TileCount * 2u);
			ShaderPrint::SetParameters(GraphBuilder, View.ShaderPrintData, MegaLightsParameters.ShaderPrintUniformBuffer);

			MegaLightsParameters.DebugLightId = CVarMegaLightsDebugLightId.GetValueOnRenderThread();

			if (MegaLightsParameters.DebugLightId < 0)
			{
				for (const FLightSceneInfoCompact& LightSceneInfoCompact : Scene->Lights)
				{
					const FLightSceneInfo* const LightSceneInfo = LightSceneInfoCompact.LightSceneInfo;

					if (LightSceneInfo->Proxy->IsSelected())
					{
						MegaLightsParameters.DebugLightId = LightSceneInfo->Id.GetIndex();
						break;
					}
				}

				if (MegaLightsParameters.DebugLightId < 0)
				{
					MegaLightsParameters.DebugLightId = InvalidDebugLightId;
				}
			}
		}
	}

	{
		extern float GInverseSquaredLightDistanceBiasScale;
		MegaLightsVolumeParameters.VolumeMinSampleWeight = FMath::Max(CVarMegaLightsVolumeMinSampleWeight.GetValueOnRenderThread(), 0.0f);
		MegaLightsVolumeParameters.VolumeDownsampleFactorMultShift = FMath::FloorLog2(VolumeDownsampleFactor);
		MegaLightsVolumeParameters.NumSamplesPerVoxel = NumSamplesPerVoxel3d;
		MegaLightsVolumeParameters.NumSamplesPerVoxelDivideShift.X = FMath::FloorLog2(NumSamplesPerVoxel3d.X);
		MegaLightsVolumeParameters.NumSamplesPerVoxelDivideShift.Y = FMath::FloorLog2(NumSamplesPerVoxel3d.Y);
		MegaLightsVolumeParameters.NumSamplesPerVoxelDivideShift.Z = FMath::FloorLog2(NumSamplesPerVoxel3d.Z);
		MegaLightsVolumeParameters.DownsampledVolumeViewSize = VolumeDownsampledViewSize;
		MegaLightsVolumeParameters.VolumeViewSize = VolumeViewSize;
		MegaLightsVolumeParameters.VolumeSampleViewSize = VolumeSampleViewSize;
		MegaLightsVolumeParameters.VolumeInvBufferSize = FVector3f(1.0f / VolumeBufferSize.X, 1.0f / VolumeBufferSize.Y, 1.0f / VolumeBufferSize.Z);
		MegaLightsVolumeParameters.MegaLightsVolumeZParams = VolumeParameters.GridZParams;
		MegaLightsVolumeParameters.MegaLightsVolumePixelSize = VolumeParameters.FogGridToPixelXY.X;
		MegaLightsVolumeParameters.VolumePhaseG = Scene->ExponentialFogs.Num() > 0 ? Scene->ExponentialFogs[0].VolumetricFogScatteringDistribution : 0.0f;
		MegaLightsVolumeParameters.VolumeInverseSquaredLightDistanceBiasScale = GInverseSquaredLightDistanceBiasScale;
		MegaLightsVolumeParameters.VolumeFrameJitterOffset = VolumetricFogTemporalRandom(View.Family->FrameNumber);
		MegaLightsVolumeParameters.UseHZBOcclusionTest = CVarMegaLightsVolumeHZBOcclusionTest.GetValueOnRenderThread();
		MegaLightsVolumeParameters.VolumeDebugSliceIndex = CVarMegaLightsDebugVolumeSliceIndex.GetValueOnRenderThread();
		MegaLightsVolumeParameters.LightSoftFading = GetVolumetricFogLightSoftFading();
		MegaLightsVolumeParameters.TranslucencyVolumeCascadeIndex = 0;
		MegaLightsVolumeParameters.TranslucencyVolumeInvResolution = 0.0f;
		MegaLightsVolumeParameters.IsUnifiedVolume = bUnifiedVolume ? 1 : 0;
		MegaLightsVolumeParameters.ResampleVolumeViewSize = VolumeViewSize;
		MegaLightsVolumeParameters.ResampleVolumeInvBufferSize = FVector3f(1.0f / VolumeBufferSize.X, 1.0f / VolumeBufferSize.Y, 1.0f / VolumeBufferSize.Z);;
		MegaLightsVolumeParameters.ResampleVolumeZParams = VolumeParameters.GridZParams;
	}

	{
		MegaLightsTranslucencyVolumeParameters.VolumeMinSampleWeight = FMath::Max(CVarMegaLightsTranslucencyVolumeMinSampleWeight.GetValueOnRenderThread(), 0.0f);
		MegaLightsTranslucencyVolumeParameters.VolumeDownsampleFactorMultShift = FMath::FloorLog2(TranslucencyVolumeDownsampleFactor);
		MegaLightsTranslucencyVolumeParameters.NumSamplesPerVoxel = NumSamplesPerTranslucencyVoxel3d;
		MegaLightsTranslucencyVolumeParameters.NumSamplesPerVoxelDivideShift.X = FMath::FloorLog2(NumSamplesPerTranslucencyVoxel3d.X);
		MegaLightsTranslucencyVolumeParameters.NumSamplesPerVoxelDivideShift.Y = FMath::FloorLog2(NumSamplesPerTranslucencyVoxel3d.Y);
		MegaLightsTranslucencyVolumeParameters.NumSamplesPerVoxelDivideShift.Z = FMath::FloorLog2(NumSamplesPerTranslucencyVoxel3d.Z);
		MegaLightsTranslucencyVolumeParameters.DownsampledVolumeViewSize = TranslucencyVolumeDownsampledViewSize;
		MegaLightsTranslucencyVolumeParameters.VolumeViewSize = TranslucencyVolumeBufferSize;
		MegaLightsTranslucencyVolumeParameters.VolumeSampleViewSize = TranslucencyVolumeSampleBufferSize;
		MegaLightsTranslucencyVolumeParameters.VolumeInvBufferSize = FVector3f(1.0f / VolumeBufferSize.X, 1.0f / VolumeBufferSize.Y, 1.0f / VolumeBufferSize.Z);
		MegaLightsTranslucencyVolumeParameters.MegaLightsVolumeZParams = FVector3f::ZeroVector;
		MegaLightsTranslucencyVolumeParameters.MegaLightsVolumePixelSize = 0;
		MegaLightsTranslucencyVolumeParameters.VolumePhaseG = 0.0f;
		MegaLightsTranslucencyVolumeParameters.VolumeInverseSquaredLightDistanceBiasScale = 1.0f;
		MegaLightsTranslucencyVolumeParameters.VolumeFrameJitterOffset = FVector3f::ZeroVector;
		MegaLightsTranslucencyVolumeParameters.UseHZBOcclusionTest = false;
		MegaLightsTranslucencyVolumeParameters.VolumeDebugSliceIndex = 0;
		MegaLightsTranslucencyVolumeParameters.LightSoftFading = 0;
		MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeCascadeIndex = 0;
		MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeInvResolution = 1.0f / GetTranslucencyLightingVolumeDim();
		MegaLightsTranslucencyVolumeParameters.IsUnifiedVolume = bUnifiedVolume ? 1 : 0;
		MegaLightsTranslucencyVolumeParameters.ResampleVolumeViewSize = VolumeViewSize;
		MegaLightsTranslucencyVolumeParameters.ResampleVolumeInvBufferSize = FVector3f(1.0f / VolumeBufferSize.X, 1.0f / VolumeBufferSize.Y, 1.0f / VolumeBufferSize.Z);;
		MegaLightsTranslucencyVolumeParameters.ResampleVolumeZParams = VolumeParameters.GridZParams;
	}

	// Build available tile types
	ShadingTileTypes = MegaLights::GetShadingTileTypes(InputType, View.GetShaderPlatform());

	ReferenceShadingPassCount = MegaLights::GetReferenceShadingPassCount(View.GetShaderPlatform());
	bReferenceMode = ReferenceShadingPassCount > 1;
	FirstPassStateFrameIndex = MegaLightsParameters.MegaLightsStateFrameIndex;
	AccumulatedRGBLightingDataFormat = bReferenceMode ? PF_A32B32G32R32F : PF_FloatRGB;
	AccumulatedRGBALightingDataFormat = bReferenceMode ? PF_A32B32G32R32F : PF_FloatRGBA;
	AccumulatedConfidenceDataFormat = bReferenceMode ? PF_R32_FLOAT : PF_R8;

	FRDGBufferRef LightPowerHistoryRatioBuffer = View.ForwardLightingResources.LightPowerHistoryRatioBuffer;

	bUseLightPowerDelta = MegaLights::UseLightPowerDelta(View) && LightPowerHistoryRatioBuffer && !bReferenceMode;

	if (!LightPowerHistoryRatioBuffer)
	{
		LightPowerHistoryRatioBuffer = GSystemTextures.GetDefaultBuffer(GraphBuilder, GPixelFormats[MegaLights::GetLightPowerDataFormat()].BlockBytes);
	}

	MegaLightsParameters.LightPowerHistoryRatioBuffer = GraphBuilder.CreateSRV(LightPowerHistoryRatioBuffer, MegaLights::GetLightPowerDataFormat());
	MegaLightsParameters.bUseLightPowerDelta = bUseLightPowerDelta ? 1 : 0;
	
	for (int32 i = 0; i < TVC_MAX; ++i)
	{
		TranslucencyVolumeResolvedLightingAmbient[i] = nullptr;
		TranslucencyVolumeResolvedLightingDirectional[i] = nullptr;
		TranslucencyVolumeVisibleLightHash[i] = nullptr;
		TranslucencyVolumeHiddenLightHash[i] = nullptr;
	}

	// Warn about this combination as it is not fully supported
	if (bUseVSM && bReferenceMode)
	{
		UE_LOGF(LogRenderer, Warning, "MegaLights Reference Mode is enabled, but VSM MegaLights are present in the scene. This setup is not fully supported and may produce artifacts!");
	}
}

void FMegaLightsViewContext::SetupStageTwo(
	FRDGTextureRef LightingChannelsTexture,
	FLumenSceneFrameTemporaries& LumenFrameTemporaries,
	ERDGPassFlags ComputePassFlags)
{
	if (MegaLights::ShouldShowLightSamplingCostHeatmap(View, InputType))
	{
		LightSamplingCostEstimate = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R16F, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("LightComplexity.LightSamplingCostEstimate"));
	}

	const FFrontLayerTranslucencyGBufferParameters FrontLayerTranslucencyGBufferParameters = GetFrontLayerTranslucencyGBufferParameters(FrontLayerTranslucencyData ? *FrontLayerTranslucencyData : FFrontLayerTranslucencyData());

	MegaLightsParameters.SceneTextures = GetSceneTextureParameters(GraphBuilder, SceneTextures.UniformBuffer);
	MegaLightsParameters.SceneTexturesStruct = CreateSceneTextureUniformBuffer(
		GraphBuilder,
		&SceneTextures,
		View.GetFeatureLevel(),
		ESceneTextureSetupMode::SceneDepth | ESceneTextureSetupMode::GBuffers);
	MegaLightsParameters.HairStrands = HairStrands::BindHairStrandsViewUniformParameters(View);
	MegaLightsParameters.LightingChannelParameters = GetSceneLightingChannelParameters(GraphBuilder, View, LightingChannelsTexture);

	if (LightingChannelsTexture)
	{
		// GetSceneLightingChannelParameters(...) only binds LightingChannelsTexture if lighting channels are enabled for the view
		// but we also need it to check HasRaytracingRepresentation(...) so always bind it here
		MegaLightsParameters.LightingChannelParameters.SceneLightingChannels = LightingChannelsTexture;
	}

	if (!MegaLights::IsUsingLightingChannels())
	{
		// LightingChannelsTexture for other reasons so force disable flag here if MegaLights is not using it
		MegaLightsParameters.LightingChannelParameters.bSceneLightingChannelsValid = false;
	}
	
	MegaLightsParameters.FrontLayerTranslucencyGBufferParameters = FrontLayerTranslucencyGBufferParameters;

	const int32 TileTypeCount = MegaLights::GetShadingTileMax(View.GetShaderPlatform());
	const int32 TileDataStride = MegaLightsParameters.TileDataStride;
	const int32 DownsampledTileDataStride = MegaLightsParameters.DownsampledTileDataStride;

	TileAllocator = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), TileTypeCount), MEGALIGHTS_RESOURCE_NAME("TileAllocator"));
	TileData = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), TileDataStride * TileTypeCount), MEGALIGHTS_RESOURCE_NAME("TileData"));
	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(TileAllocator), 0, ComputePassFlags);

	DownsampledTileAllocator = TileAllocator;
	DownsampledTileData = TileData;

	if (DownsampleFactor.X != 1)
	{
		DownsampledTileAllocator = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), TileTypeCount), MEGALIGHTS_RESOURCE_NAME("DownsampledTileAllocator"));
		DownsampledTileData = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), DownsampledTileDataStride * TileTypeCount), MEGALIGHTS_RESOURCE_NAME("DownsampledTileData"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(DownsampledTileAllocator), 0, ComputePassFlags);
	}

	// Run tile classification to generate tiles for the subsequent passes
	{
		HistoryScreenParameters = StochasticLighting::GetHistoryScreenParameters(View);

		if (InputType == EMegaLightsInput::GBuffer)
		{
			// Opaque was already tile classified
			MegaLightsParameters.MegaLightsTileBitmask = LumenFrameTemporaries.StochasticLighting.MegaLightsTileBitmask.GetRenderTarget();
			HistoryScreenParameters.EncodedHistoryScreenCoordTexture = LumenFrameTemporaries.StochasticLighting.EncodedHistoryScreenCoord.GetRenderTarget();
			HistoryScreenParameters.PackedPixelDataTexture = LumenFrameTemporaries.StochasticLighting.MegaLightsPackedPixelData.GetRenderTarget();
			
			// DepthHistory/NormalHistory are copies of current frame depth/normal and will be used as history next frame
			SceneWorldNormal = LumenFrameTemporaries.StochasticLighting.NormalHistory.GetRenderTarget();
			MegaLightsVisualizeParameters.OpaqueOnlyDepthTexture = LumenFrameTemporaries.StochasticLighting.DepthHistory.GetRenderTarget();
		}
		else if (InputType == EMegaLightsInput::HairStrands)
		{
			TileClassificationMark(0 /*ShadingPassIndex*/, ComputePassFlags);
		}
		else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
		{
			check(FrontLayerTranslucencyData);

			// FrontLayerTranslucency was already tile classified
			MegaLightsParameters.MegaLightsTileBitmask = FrontLayerTranslucencyData->StochasticLighting.MegaLightsTileBitmask.GetRenderTarget();
			HistoryScreenParameters.EncodedHistoryScreenCoordTexture = FrontLayerTranslucencyData->StochasticLighting.EncodedHistoryScreenCoord.GetRenderTarget();
			HistoryScreenParameters.PackedPixelDataTexture = FrontLayerTranslucencyData->StochasticLighting.MegaLightsPackedPixelData.GetRenderTarget();

			// Just use front layer translucency GBufferA since we don't output NormalHistory during front layer's tile classification mark
			SceneWorldNormal = FrontLayerTranslucencyData->GBufferA;
		}
		else
		{
			checkNoEntry();
		}

		{
			FMegaLightsTileClassificationBuildListsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsTileClassificationBuildListsCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->MegaLightsParameters.TileDataStride = TileDataStride;
			PassParameters->RWTileAllocator = GraphBuilder.CreateUAV(TileAllocator);
			PassParameters->RWTileData = GraphBuilder.CreateUAV(TileData);
			PassParameters->ViewSizeInTiles = ViewSizeInTiles;
			PassParameters->ViewMinInTiles = FMath::DivideAndRoundUp<FIntPoint>(View.ViewRect.Min, MegaLights::TileSize);
			PassParameters->DownsampledViewSizeInTiles = DownsampledViewSizeInTiles;
			PassParameters->DownsampledViewMinInTiles = FMath::DivideAndRoundUp<FIntPoint>(MegaLightsParameters.DownsampledViewMin, MegaLights::TileSize);
			PassParameters->OutputTileDataStride = TileDataStride;
			PassParameters->UseTexturedRectLights = MegaLights::UseTexturedRectLights(InputType) ? 1 : 0;

			FMegaLightsTileClassificationBuildListsCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FDownsampleFactorX>(1);
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FDownsampleFactorY>(1);
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FFastClear>(MegaLights::UseFastClear(InputType));
			PermutationVector = FMegaLightsTileClassificationBuildListsCS::RemapPermutation(PermutationVector);
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsTileClassificationBuildListsCS>(PermutationVector);
				
			const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(ViewSizeInTiles, FMegaLightsTileClassificationBuildListsCS::GetGroupSize());

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("TileClassificationBuildLists"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				GroupCount);
		}

		if (DownsampleFactor.X != 1)
		{
			FMegaLightsTileClassificationBuildListsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsTileClassificationBuildListsCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->RWTileAllocator = GraphBuilder.CreateUAV(DownsampledTileAllocator);
			PassParameters->RWTileData = GraphBuilder.CreateUAV(DownsampledTileData);
			PassParameters->ViewSizeInTiles = ViewSizeInTiles;
			PassParameters->ViewMinInTiles = FMath::DivideAndRoundUp<FIntPoint>(View.ViewRect.Min, MegaLights::TileSize);
			PassParameters->DownsampledViewSizeInTiles = DownsampledViewSizeInTiles;
			PassParameters->DownsampledViewMinInTiles = FMath::DivideAndRoundUp<FIntPoint>(MegaLightsParameters.DownsampledViewMin, MegaLights::TileSize);
			PassParameters->OutputTileDataStride = DownsampledTileDataStride;
			PassParameters->UseTexturedRectLights = MegaLights::UseTexturedRectLights(InputType) ? 1 : 0;

			FMegaLightsTileClassificationBuildListsCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FDownsampleFactorX>(DownsampleFactor.X);
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FDownsampleFactorY>(DownsampleFactor.Y);
			PermutationVector.Set<FMegaLightsTileClassificationBuildListsCS::FFastClear>(MegaLights::UseFastClear(InputType));
			PermutationVector = FMegaLightsTileClassificationBuildListsCS::RemapPermutation(PermutationVector);
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsTileClassificationBuildListsCS>(PermutationVector);

			const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(DownsampledViewSizeInTiles, FMegaLightsTileClassificationBuildListsCS::GetGroupSize());

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("DownsampledTileClassificationBuildLists"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				GroupCount);
		}
	}

	TileIndirectArgs = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>(TileTypeCount), MEGALIGHTS_RESOURCE_NAME("TileIndirectArgs"));
	DownsampledTileIndirectArgs = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>(TileTypeCount), MEGALIGHTS_RESOURCE_NAME("DownsampledTileIndirectArgs"));

	// Setup indirect args for classified tiles
	{
		FInitTileIndirectArgsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FInitTileIndirectArgsCS::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->RWTileIndirectArgs = GraphBuilder.CreateUAV(TileIndirectArgs);
		PassParameters->RWDownsampledTileIndirectArgs = GraphBuilder.CreateUAV(DownsampledTileIndirectArgs);
		PassParameters->TileAllocator = GraphBuilder.CreateSRV(TileAllocator);
		PassParameters->DownsampledTileAllocator = GraphBuilder.CreateSRV(DownsampledTileAllocator);

		auto ComputeShader = View.ShaderMap->GetShader<FInitTileIndirectArgsCS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("InitTileIndirectArgs"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FIntVector(1, 1, 1));
	}
}

void FMegaLightsViewContext::MarkVSMPages(
	const FVirtualShadowMapArray& VirtualShadowMapArray)
{
	if (bUseVSM && MegaLights::IsMarkingVSMPages())
	{
		if (InputType == EMegaLightsInput::GBuffer)
		{
			MegaLights::MarkVSMPages(
				View, ViewIndex,
				GraphBuilder,
				VirtualShadowMapArray,
				SampleBufferSize,
				LightSamples,
				LightSampleRays,
				MegaLightsParameters,
				InputType);
		}
		else if (InputType == EMegaLightsInput::HairStrands)
		{
			// TODO: VSM marking for hair strands
			UE_LOGF(LogRenderer, Warning, "MegaLights VSM marking is not yet implemented for HairStrands. Disable with r.MegaLights.VSM.MarkPages.");
		}
		else if (InputType == EMegaLightsInput::FrontLayerTranslucency)
		{
			unimplemented();
		}
		else
		{
			checkNoEntry();
		}
	}
}

void FMegaLightsViewContext::RayTrace(
	const FVirtualShadowMapArray& VirtualShadowMapArray,
	const FBoxSphereBounds& FirstPersonWorldSpaceRepresentationBounds,
	uint32 ShadingPassIndex,
	ERDGPassFlags ComputePassFlags)
{
	const bool bDebugPass = (bDebug || MegaLights::IsVisualizeRaysEnabled()) && MegaLights::IsDebugEnabledForShadingPass(ShadingPassIndex, View.GetShaderPlatform());
	const bool bTraceVolumeRays = !bVolumeRaysTraced || ShadingPassIndex > 0;

	MegaLights::FVolumeLightSampleParameters VolumeLightSampleParameters;
	if (bTraceVolumeRays)
	{
		VolumeLightSampleParameters.VolumeSampleBufferSize = VolumeSampleBufferSize;
		VolumeLightSampleParameters.VolumeLightSamples = VolumeLightSamples;
		VolumeLightSampleParameters.VolumeLightSampleRays = VolumeLightSampleRays;
		VolumeLightSampleParameters.TranslucencyVolumeSampleBufferSize = TranslucencyVolumeSampleBufferSize;
		VolumeLightSampleParameters.TranslucencyVolumeLightSamples = TranslucencyVolumeLightSamples;
		VolumeLightSampleParameters.TranslucencyVolumeLightSampleRays = TranslucencyVolumeLightSampleRays;
	}

	MegaLights::RayTraceLightSamples(
		ViewFamily,
		View, ViewIndex,
		GraphBuilder,
		SceneTextures,
		bUseVSM ? &VirtualShadowMapArray : nullptr,
		FirstPersonWorldSpaceRepresentationBounds,
		SampleBufferSize,
		LightSamples,
		LightSampleRays,
		MegaLightsParameters,
		MegaLightsVolumeParameters,
		MegaLightsTranslucencyVolumeParameters,
		bTraceVolumeRays ? &VolumeLightSampleParameters : nullptr,
		bTraceVolumeRays ? nullptr : &VolumeTraceStats,
		InputType,
		bDebugPass,
		ComputePassFlags
	);

	bVolumeRaysTraced = true;
}

void FMegaLightsViewContext::VolumeRayTrace(ERDGPassFlags ComputePassFlags)
{
	check(!bVolumeRaysTraced);

	MegaLights::FVolumeLightSampleParameters VolumeLightSampleParameters;
	VolumeLightSampleParameters.VolumeSampleBufferSize = VolumeSampleBufferSize;
	VolumeLightSampleParameters.VolumeLightSamples = VolumeLightSamples;
	VolumeLightSampleParameters.VolumeLightSampleRays = VolumeLightSampleRays;
	VolumeLightSampleParameters.TranslucencyVolumeSampleBufferSize = TranslucencyVolumeSampleBufferSize;
	VolumeLightSampleParameters.TranslucencyVolumeLightSamples = TranslucencyVolumeLightSamples;
	VolumeLightSampleParameters.TranslucencyVolumeLightSampleRays = TranslucencyVolumeLightSampleRays;

	VolumeTraceStats = MegaLights::RayTraceVolumeLightSamples(
		GraphBuilder,
		View,
		ViewFamily,
		MegaLightsParameters,
		MegaLightsVolumeParameters,
		MegaLightsTranslucencyVolumeParameters,
		VolumeLightSampleParameters,
		nullptr /*CompactedTraceParameters*/,
		ComputePassFlags);

	bVolumeRaysTraced = true;
}

bool FMegaLightsViewContext::IsAsyncComputeAllowed() const
{
	return !bUseVSM && !bReferenceMode;
}

bool FMegaLightsViewContext::VolumeUseAsyncCompute() const
{
	return IsAsyncComputeAllowed()
		&& CVarMegaLightsAsyncComputeVolume.GetValueOnRenderThread() != 0;
}

bool FMegaLightsViewContext::GenerateSamplesUseAsyncCompute() const
{
	return IsAsyncComputeAllowed()
		&& CVarMegaLightsAsyncComputeGenerateSamples.GetValueOnRenderThread() != 0;
}

void FMegaLightsViewContext::SetFrontLayerTranslucencyData(const FFrontLayerTranslucencyData& InFrontLayerTranslucencyData)
{
	FrontLayerTranslucencyData = &InFrontLayerTranslucencyData;
}

bool FMegaLightsViewContext::HasFrontLayerTranslucencyData() const
{
	return FrontLayerTranslucencyData != nullptr && FrontLayerTranslucencyData->IsValid();
}

void FMegaLightsViewContext::UpdateHZB()
{
	MegaLightsParameters.HZBParameters.HZBTexture = View.HZB;
	MegaLightsParameters.HZBParameters.ClosestHZBTexture = View.ClosestHZB;
	MegaLightsParameters.HZBParameters.FurthestHZBTexture = View.HZB;
}

void FMegaLightsFrameTemporaries::UpdateHZB()
{
	for (FMegaLightsViewContext& Context : ViewContexts)
	{
		Context.UpdateHZB();
	}
}

void FMegaLightsViewState::FResources::SafeRelease()
{
	DiffuseLightingHistory.SafeRelease();
	SpecularLightingHistory.SafeRelease();
	LightingMomentsHistory.SafeRelease();
	NumFramesAccumulatedHistory.SafeRelease();
	VisibleLightHashHistory.SafeRelease();
	HiddenLightHashHistory.SafeRelease();
	VolumeVisibleLightHashHistory.SafeRelease();
	VolumeHiddenLightHashHistory.SafeRelease();
	TranslucencyVolume0VisibleLightHashHistory.SafeRelease();
	TranslucencyVolume1VisibleLightHashHistory.SafeRelease();
	TranslucencyVolume0HiddenLightHashHistory.SafeRelease();
	TranslucencyVolume1HiddenLightHashHistory.SafeRelease();
	FrozenLightData.SafeRelease();
	GeneralPurposeState.SafeRelease();
	if (DumpLightDataReadback)
	{
		delete DumpLightDataReadback;
		DumpLightDataReadback = nullptr;
	}
	if (GeneralPurposeReadback)
	{
		delete GeneralPurposeReadback;
		GeneralPurposeReadback = nullptr;
	}
}

bool MegaLights::UseTranslucencyVolumeMarkTexture(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
{
	if (!MegaLightsFrameTemporaries)
	{
		return false;
	}

	const bool bView0UseMarkTexture = IsTranslucencyLightingVolumeUsingVoxelMarking()
		&& MegaLightsFrameTemporaries->ViewContexts.Num() > 0
		&& MegaLightsFrameTemporaries->ViewContexts[0].ShouldRenderTranslucencyVolume();

	for (int32 ViewIndex = 1; ViewIndex < MegaLightsFrameTemporaries->ViewContexts.Num(); ++ViewIndex)
	{
		check(MegaLightsFrameTemporaries->ViewContexts[0].ShouldRenderTranslucencyVolume() == MegaLightsFrameTemporaries->ViewContexts[ViewIndex].ShouldRenderTranslucencyVolume());
	}

	return bView0UseMarkTexture;
}

bool MegaLights::VolumeUseAsyncCompute(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
{
	if (!MegaLightsFrameTemporaries)
	{
		return false;
	}

	const bool bView0UseAsyncCompute = MegaLightsFrameTemporaries->ViewContexts.Num() > 0
		&& MegaLightsFrameTemporaries->ViewContexts[0].VolumeUseAsyncCompute();

	for (int32 ViewIndex = 1; ViewIndex < MegaLightsFrameTemporaries->ViewContexts.Num(); ++ViewIndex)
	{
		check(bView0UseAsyncCompute == MegaLightsFrameTemporaries->ViewContexts[ViewIndex].VolumeUseAsyncCompute());
	}

	return bView0UseAsyncCompute;
}

bool MegaLights::GenerateSamplesUseAsyncCompute(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
{
	if (!MegaLightsFrameTemporaries)
	{
		return false;
	}

	const bool bView0UseAsyncCompute = MegaLightsFrameTemporaries->ViewContexts.Num() > 0
		&& MegaLightsFrameTemporaries->ViewContexts[0].GenerateSamplesUseAsyncCompute();

	for (int32 ViewIndex = 1; ViewIndex < MegaLightsFrameTemporaries->ViewContexts.Num(); ++ViewIndex)
	{
		check(bView0UseAsyncCompute == MegaLightsFrameTemporaries->ViewContexts[ViewIndex].GenerateSamplesUseAsyncCompute());
	}

	return bView0UseAsyncCompute;
}

TSharedPtr<FMegaLightsFrameTemporaries> FDeferredShadingSceneRenderer::CreateMegaLightsFrameTemporaries(FRDGBuilder& GraphBuilder, const FSceneTextures& SceneTextures)
{
	if (!MegaLights::IsEnabled(ViewFamily))
	{
		return nullptr;
	}

	check(AreLightsInLightGrid());
	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLights, "MegaLights");

	TSharedPtr<FMegaLightsFrameTemporaries> MegaLightsFrameTemporaries(new FMegaLightsFrameTemporaries);

	FShadowSceneRenderer& ShadowSceneRenderer = GetSceneExtensionsRenderers().GetRenderer<FShadowSceneRenderer>();
	const bool bUseVSM = ShadowSceneRenderer.AreAnyLightsUsingMegaLightsVSM();

	FBlueNoise BlueNoise = GetBlueNoiseGlobalParameters();
	MegaLightsFrameTemporaries->BlueNoiseUniformBuffer = CreateUniformBufferImmediate(BlueNoise, EUniformBufferUsage::UniformBuffer_SingleDraw);

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];

		FMegaLightsViewContext InitContext(
			GraphBuilder,
			ViewIndex,
			View,
			ViewFamily,
			Scene,
			SceneTextures,
			bUseVSM);

		FMegaLightsViewContext& ViewContext = MegaLightsFrameTemporaries->ViewContexts.Add_GetRef(InitContext);
		FMegaLightsViewContext& ViewContextHairStrands = MegaLightsFrameTemporaries->ViewContextsHairStrands.Add_GetRef(InitContext);
		FMegaLightsViewContext& ViewContextFrontLayerTranslucency = MegaLightsFrameTemporaries->ViewContextsFrontLayerTranslucency.Add_GetRef(InitContext);

		ViewContext.SetupStageOne(
			ShouldRenderVolumetricFog(),
			GUseTranslucencyLightingVolumes && MegaLights::UseTranslucencyVolume(),
			MegaLightsFrameTemporaries->BlueNoiseUniformBuffer,
			EMegaLightsInput::GBuffer);

		// Hair Strands context cannot be initialized early because we only know whether data is valid after hair base pass

		// Front Layer Translucency context cannot be initialized early because we only render front layer translucency GBuffer later in the frame
	}

	return MegaLightsFrameTemporaries;
}

void FDeferredShadingSceneRenderer::DispatchAsyncMegaLightsVolumePasses(
	FRDGBuilder& GraphBuilder,
	const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
{
	if (!MegaLights::VolumeUseAsyncCompute(MegaLightsFrameTemporaries))
	{
		return;
	}

	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLights, "MegaLights");

	const ERDGPassFlags ComputePassFlags = ERDGPassFlags::AsyncCompute;

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];
		FMegaLightsViewContext& ViewContext = MegaLightsFrameTemporaries->ViewContexts[ViewIndex];
		check(!ViewContext.AreVolumeSamplesGenerated());

		ViewContext.GenerateVolumeSamples(ComputePassFlags);
		ViewContext.VolumeRayTrace(ComputePassFlags);
		View.GetOwnMegaLightsVolume() = ViewContext.VolumeResolve(ComputePassFlags);
	}
}

void FDeferredShadingSceneRenderer::GenerateMegaLightsSamples(
	FRDGBuilder& GraphBuilder,
	const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries,
	FLumenSceneFrameTemporaries& LumenFrameTemporaries,
	FRDGTextureRef LightingChannelsTexture,
	ERDGPassFlags ComputePassFlags)
{
	if (!MegaLightsFrameTemporaries)
	{
		return;
	}

	check(AreLightsInLightGrid());
	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLights, "MegaLights");

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];
		const bool bHairStrands = MegaLights::IsHairStrandsEnabled(View);
		
		FMegaLightsViewContext& ViewContext = MegaLightsFrameTemporaries->ViewContexts[ViewIndex];
		FMegaLightsViewContext& ViewContextHairStrands = MegaLightsFrameTemporaries->ViewContextsHairStrands[ViewIndex];

		{
			RDG_EVENT_SCOPE_CONDITIONAL(GraphBuilder, bHairStrands, "GBuffer");

			ViewContext.SetupStageTwo(LightingChannelsTexture, LumenFrameTemporaries, ComputePassFlags);

			ViewContext.GenerateSamples(LightingChannelsTexture, 0 /* ShadingPassIndex */, ComputePassFlags);

			if (!ViewContext.AreVolumeSamplesGenerated())
			{
				ViewContext.GenerateVolumeSamples(ComputePassFlags);
			}

			// Mark VSM pages for any required samples
			ViewContext.MarkVSMPages(VirtualShadowMapArray);
		}

		if (bHairStrands)
		{
			RDG_EVENT_SCOPE_CONDITIONAL(GraphBuilder, bHairStrands, "HairStrands");

			ViewContextHairStrands.SetupStageOne(
				false /*bShouldRenderVolumetricFog*/,
				false /*bShouldRenderTranslucencyVolume*/,
				MegaLightsFrameTemporaries->BlueNoiseUniformBuffer,
				EMegaLightsInput::HairStrands);

			ViewContextHairStrands.SetupStageTwo(LightingChannelsTexture, LumenFrameTemporaries, ComputePassFlags);

			ViewContextHairStrands.GenerateSamples(LightingChannelsTexture, 0 /* ShadingPassIndex */, ComputePassFlags);

			ViewContextHairStrands.MarkVSMPages(VirtualShadowMapArray);
		}
	}

	GMegaLightsVisualizeLightComplexityDump = 0;
}

static void RenderMegaLightsViewContext(
	FRDGBuilder& GraphBuilder,
	FMegaLightsViewContext& ViewContext,
	const FVirtualShadowMapArray& VirtualShadowMapArray,
	const FBoxSphereBounds& FirstPersonWorldSpaceRepresentationBounds,
	FRDGTextureRef LightingChannelsTexture,
	FMegaLightsVolume* MegaLightsVolume,
	FRDGTextureRef OutputColorTarget,
	ERDGPassFlags ComputePassFlags)
{
	check(ViewContext.AreSamplesGenerated());

	// In reference mode we loop over the raytracing and sample generation
	// NOTE: This does not work properly with MegaLights VSM marking as we would need
	// to go back and mark any new samples, then potentially render new shadow maps for
	// those samples as well, but this mode is designed to be used with high quality raytracing.
	for (uint32 ShadingPassIndex = 0; ShadingPassIndex < ViewContext.GetReferenceShadingPassCount(); ++ShadingPassIndex)
	{
		// We've already generated sample 0 separately, but following passes need new samples
		if (ShadingPassIndex > 0)
		{
			ViewContext.TileClassificationMark(ShadingPassIndex, ComputePassFlags);

			ViewContext.GenerateSamples(LightingChannelsTexture, ShadingPassIndex, ComputePassFlags);

			ViewContext.GenerateVolumeSamples(ComputePassFlags);
		}
						
		ViewContext.RayTrace(
			VirtualShadowMapArray,
			FirstPersonWorldSpaceRepresentationBounds,
			ShadingPassIndex,
			ComputePassFlags);

		ViewContext.Resolve(
			OutputColorTarget,
			MegaLightsVolume,
			ShadingPassIndex,
			ComputePassFlags);
	}

	ViewContext.DispatchDebugTileClassificationPasses(ComputePassFlags);
	ViewContext.DispatchVisualizeLightsPasses(ComputePassFlags);

	ViewContext.DenoiseLighting(OutputColorTarget, ComputePassFlags);
}

void FDeferredShadingSceneRenderer::RenderMegaLights(
	FRDGBuilder& GraphBuilder,
	const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries,
	const FSceneTextures& SceneTextures,
	FRDGTextureRef LightingChannelsTexture)
{
	if (!MegaLightsFrameTemporaries.IsValid())
	{
		return;
	}

	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLights, "MegaLights");

	for (int32 ViewIndex = 0; ViewIndex < MegaLightsFrameTemporaries->ViewContexts.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];
		FMegaLightsViewContext& ViewContext = MegaLightsFrameTemporaries->ViewContexts[ViewIndex];
		FMegaLightsViewContext& ViewContextHairStrands = MegaLightsFrameTemporaries->ViewContextsHairStrands[ViewIndex];
		const bool bHairStrands = ViewContextHairStrands.AreSamplesGenerated();

		FBoxSphereBounds FirstPersonWorldSpaceRepresentationBounds = FBoxSphereBounds(ForceInit);
		if (const FFirstPersonSceneExtensionRenderer* FPRenderer = GetSceneExtensionsRenderers().GetRendererPtr<FFirstPersonSceneExtensionRenderer>())
		{
			FirstPersonWorldSpaceRepresentationBounds = FPRenderer->GetFirstPersonViewBounds(View).WorldSpaceRepresentationBounds;
		}
		
		{
			RDG_EVENT_SCOPE_CONDITIONAL(GraphBuilder, bHairStrands, "GBuffer");

			RenderMegaLightsViewContext(
				GraphBuilder,
				ViewContext,
				VirtualShadowMapArray,
				FirstPersonWorldSpaceRepresentationBounds,
				LightingChannelsTexture,
				&View.GetOwnMegaLightsVolume(),
				SceneTextures.Color.Target,
				ERDGPassFlags::Compute);
		}

		if (bHairStrands)
		{
			RDG_EVENT_SCOPE_CONDITIONAL(GraphBuilder, bHairStrands, "HairStrands");

			RenderMegaLightsViewContext(
				GraphBuilder, 
				ViewContextHairStrands,
				VirtualShadowMapArray,
				FirstPersonWorldSpaceRepresentationBounds,
				LightingChannelsTexture, 
				nullptr /*MegaLightsVolume*/, 
				View.HairStrandsViewData.VisibilityData.SampleLightingTexture,
				ERDGPassFlags::Compute);
		}
	}
}

void FDeferredShadingSceneRenderer::RenderMegaLightsFrontLayerTranslucency(
	FRDGBuilder& GraphBuilder,
	TSharedPtr<FMegaLightsFrameTemporaries> MegaLightsFrameTemporaries,
	FLumenSceneFrameTemporaries& LumenFrameTemporaries,
	const FFrontLayerTranslucencyData& FrontLayerTranslucencyData,
	ERDGPassFlags ComputePassFlags)
{
	if (!MegaLightsFrameTemporaries.IsValid() || !FrontLayerTranslucencyData.IsValid())
	{
		return;
	}

	RDG_EVENT_SCOPE_STAT(GraphBuilder, MegaLightsFrontLayerTranslucency, "MegaLightsFrontLayerTranslucency");

	FRDGTextureRef DirectLighting = GraphBuilder.CreateTexture(
		FRDGTextureDesc::Create2D(GetActiveSceneTexturesConfig().Extent, MegaLights::GetLightingDataFormat(), FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
		TEXT("MegaLights.FrontLayerTranslucency"));

	for (int32 ViewIndex = 0; ViewIndex < MegaLightsFrameTemporaries->ViewContexts.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];

		if (!MegaLights::UseFrontLayerTranslucencyDirectLighting(View))
		{
			continue;
		}

		FMegaLightsViewContext& ViewContextFrontLayerTranslucency = MegaLightsFrameTemporaries->ViewContextsFrontLayerTranslucency[ViewIndex];

		FBoxSphereBounds FirstPersonWorldSpaceRepresentationBounds = FBoxSphereBounds(ForceInit);
		if (const FFirstPersonSceneExtensionRenderer* FPRenderer = GetSceneExtensionsRenderers().GetRendererPtr<FFirstPersonSceneExtensionRenderer>())
		{
			FirstPersonWorldSpaceRepresentationBounds = FPRenderer->GetFirstPersonViewBounds(View).WorldSpaceRepresentationBounds;
		}

		ViewContextFrontLayerTranslucency.SetFrontLayerTranslucencyData(FrontLayerTranslucencyData);

		ViewContextFrontLayerTranslucency.SetupStageOne(
			false /*bShouldRenderVolumetricFog*/,
			false /*bShouldRenderTranslucencyVolume*/,
			MegaLightsFrameTemporaries->BlueNoiseUniformBuffer,
			EMegaLightsInput::FrontLayerTranslucency);

		ViewContextFrontLayerTranslucency.SetupStageTwo(nullptr, LumenFrameTemporaries, ComputePassFlags);

		ViewContextFrontLayerTranslucency.GenerateSamples(nullptr, /* ShadingPassIndex */ 0, ComputePassFlags);

		View.FrontLayerTranslucency.bDirectLightingEnabled = true;
		View.FrontLayerTranslucency.bSpecularOnly = CVarMegaLightsFrontLayerTranslucencySpecularOnly.GetValueOnRenderThread();

		View.FrontLayerTranslucency.DirectLighting = DirectLighting;

		RenderMegaLightsViewContext(
			GraphBuilder,
			ViewContextFrontLayerTranslucency,
			VirtualShadowMapArray,
			FirstPersonWorldSpaceRepresentationBounds,
			nullptr,
			nullptr /*MegaLightsVolume*/,
			View.FrontLayerTranslucency.DirectLighting,
			ComputePassFlags);
	}
}

namespace MegaLights
{
	bool IsMissingDirectionalLightData(const FSceneViewFamily& ViewFamily)
	{
		static auto LightBufferModeCVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.Forward.LightBuffer.Mode"));
		
		return CVarMegaLightsDirectionalLights.GetValueOnRenderThread() && LightBufferModeCVar->GetInt() == 0;
	}

	bool IsViewVisualizingMegaLights(const FViewInfo& View)
	{
		if (MegaLights::GetVisualizeLightComplexityTarget(View) != EMegaLightsDebugTarget::None)
		{
			return true;
		}

		if (MegaLights::GetVisualizeMode(View) > 0)
		{
			return true;
		}

		return false;
	}

	bool IsViewVisualizingMegaLightsFrontLayerTranslucency(const FViewInfo& View)
	{
		if (MegaLights::GetVisualizeLightComplexityTarget(View) == EMegaLightsDebugTarget::FrontLayerTranslucency)
		{
			return true;
		}

		return false;
	}

	bool HasWarning(const FViewInfo& View)
	{
		const bool bRequested = IsRequested(*View.Family);

		if (bRequested && (!HasRequiredTracingData(*View.Family) || IsMissingDirectionalLightData(*View.Family)))
		{
			return true;
		}

		if (!bRequested && IsViewVisualizingMegaLights(View))
		{
			return true;
		}

		if (bRequested && !UseFrontLayerTranslucencyDirectLighting(View) && IsViewVisualizingMegaLightsFrontLayerTranslucency(View))
		{
			return true;
		}

		return false;
	}

	void WriteWarnings(const FViewInfo& View, FScreenMessageWriter& Writer)
	{
		if (!HasWarning(View))
		{
			return;
		}

		const bool bRequested = IsRequested(*View.Family);

		if (bRequested && !HasRequiredTracingData(*View.Family))
		{
			static const FText MainMessage = NSLOCTEXT("Renderer", "MegaLightsCantDisplay", "MegaLights is enabled, but has no ray tracing data and won't operate correctly.");
			Writer.DrawLine(MainMessage);

#if RHI_RAYTRACING
			if (!IsRayTracingAllowed())
			{
				static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToHWRTNotAllowed", "- Hardware Ray Tracing is not allowed. Check log for more info.");
				Writer.DrawLine(Message);
			}
			else if (!IsRayTracingEnabled())
			{
				static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToHWRTDisabled", "- Enable 'r.RayTracing.Enable'.");
				Writer.DrawLine(Message);
			}

			static auto CVarMegaLightsHardwareRayTracing = IConsoleManager::Get().FindConsoleVariable(TEXT("r.MegaLights.HardwareRayTracing"));
			if (CVarMegaLightsHardwareRayTracing->GetInt() == 0)
			{
				static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToCvar", "- Enable 'r.MegaLights.HardwareRayTracing'.");
				Writer.DrawLine(Message);
			}

			static auto CVarMegaLightsHardwareRayTracingInline = IConsoleManager::Get().FindConsoleVariable(TEXT("r.MegaLights.HardwareRayTracing.Inline"));
			if (!(GRHISupportsRayTracingShaders || (GRHISupportsInlineRayTracing && CVarMegaLightsHardwareRayTracingInline->GetInt() != 0)))
			{
				static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToPlatformSettings", "- Enable Full Ray Tracing in platform platform settings or r.MegaLights.HardwareRayTracing.Inline.");
				Writer.DrawLine(Message);
			}

			if (!View.Family->Views[0]->IsRayTracingAllowedForView())
			{
				static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToView", "- Ray Tracing not allowed on the View.");
				Writer.DrawLine(Message);
			}
#else
			static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDueToBuild", "- Unreal Engine was built without Hardware Ray Tracing support.");
			Writer.DrawLine(Message);
#endif
		}

		if (bRequested && IsMissingDirectionalLightData(*View.Family))
		{
			static const FText MainMessage = NSLOCTEXT("Renderer", "MegaLightsCantDisplayDirectionalLights", "MegaLights requires r.Forward.LightBuffer.Mode > 0 when using r.MegaLights.DirectionalLights=1.");
			Writer.DrawLine(MainMessage);
		}

		if (!bRequested && IsViewVisualizingMegaLights(View))
		{
			static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantVisualize", "MegaLights visualization is enabled but cannot render, because MegaLights is disabled. Check PPV setting, r.MegaLights.Allowed, showflags, and platform support.");
			Writer.DrawLine(Message);
		}

		if (bRequested && !UseFrontLayerTranslucencyDirectLighting(View) && IsViewVisualizingMegaLightsFrontLayerTranslucency(View))
		{
			static const FText Message = NSLOCTEXT("Renderer", "MegaLightsCantVisualizeFrontLayerTranslucency", 
				"MegaLights Front Layer Translucency visualization is enabled but cannot render, because the feature is disabled. Check PPV setting, project setting, r.MegaLights.FrontLayerTranslucency.Allowed, showflags, and platform support.");
			Writer.DrawLine(Message);
		}
	}
}