// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreTypes.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RHIGPUReadback.h"
#include "RHICommandList.h"
#include "SceneManagement.h"                  // FWorkingColorSpaceShaderParameters, GDefaultWorkingColorSpaceUniformBuffer
#include "ColorManagement/ColorSpace.h"
#include "PostProcess/ACESUtils.h"            // UE::Color::ACES::GetTransformResources
#include "HAL/IConsoleManager.h"
#include "GenericPlatform/GenericPlatformProcess.h"

namespace UE::Renderer::Private::ACES2Test
{

// Matches the test shader's [numthreads(THREADGROUP_SIZE,1,1)].
static constexpr int32 ThreadGroupSize = 64;

// SDR sRGB configuration, matching the default CombineLUTs path (see GetTransformResourcesSDR).
static constexpr float  SDRPeakLuminance = 100.0f;   // nits
static constexpr uint32 OutputGamut_sRGB_D65 = 0;    // TONEMAPPER_GAMUT_sRGB_D65

/** Test-only compute shader that runs the production ACES 2.0 forward + inverse transforms. */
class FACESOutputTransformTestCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FACESOutputTransformTestCS);
	SHADER_USE_PARAMETER_STRUCT(FACESOutputTransformTestCS, FGlobalShader);

	// false: scene -> forward -> display -> inverse -> scene (OutForward = nits, OutRoundTrip = recovered scene).
	// true:  display -> inverse -> scene -> forward -> display (OutForward = scene, OutRoundTrip = recovered display).
	class FDisplaySideRoundTrip : SHADER_PERMUTATION_BOOL("DISPLAY_SIDE_ROUNDTRIP");
	using FPermutationDomain = TShaderPermutationDomain<FDisplaySideRoundTrip>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FWorkingColorSpaceShaderParameters, WorkingColorSpace)
		SHADER_PARAMETER_SRV(Texture2D<float>, ACESReachTable)
		SHADER_PARAMETER_SRV(Texture2D<float>, ACESGamutTable)
		SHADER_PARAMETER_SRV(Texture2D<float>, ACESGammaTable)
		SHADER_PARAMETER(FMatrix44f, LimitingRgbToXYZ)
		SHADER_PARAMETER(FMatrix44f, LimitingXYZToRgb)
		SHADER_PARAMETER(uint32, bScaleWhite)
		SHADER_PARAMETER(uint32, OutputGamut)
		SHADER_PARAMETER(float, PeakLuminance)
		SHADER_PARAMETER(uint32, NumColors)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>, InputColors)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float4>, OutForward)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float4>, OutRoundTrip)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		// Tests don't currently run on mobile platforms, so skip compiling those permutations.
		return !IsMobilePlatform(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("USE_ACES_2"), 1);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
	}
};

IMPLEMENT_GLOBAL_SHADER(FACESOutputTransformTestCS, "/Engine/Private/ACES/ACESOutputTransformTest.usf", "MainCS", SF_Compute);

} // namespace UE::Renderer::Private::ACES2Test

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FACESOutputTransformTest, "System.Renderer.ColorManagement.ACES2OutputTransform",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter | EAutomationTestFlags::NonNullRHI)

bool FACESOutputTransformTest::RunTest(const FString& Parameters)
{
	using namespace UE::Renderer::Private::ACES2Test;

	// --- Precondition: force ACES 2.0 on, else GetTransformResources returns black tables (ACES 1.3 fallback). ---
	IConsoleVariable* AcesVersionCVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.HDR.Aces.Version"));
	UTEST_TRUE(TEXT("r.HDR.Aces.Version CVar exists"), AcesVersionCVar != nullptr);
	// UTEST_TRUE already returns on null, but the static analyzer can't see through the macro; hint it (C6011).
	CA_ASSUME(AcesVersionCVar != nullptr);

	const int32 SavedAcesVersion = AcesVersionCVar->GetInt();
	AcesVersionCVar->Set(2, ECVF_SetByCode);
	// Restore on every exit path.
	ON_SCOPE_EXIT
	{
		AcesVersionCVar->Set(SavedAcesVersion, ECVF_SetByCode);
	};
	// Flush so the render-thread shadow value (read via GetValueOnRenderThread) is updated before our pass runs.
	FlushRenderingCommands();

	// --- Prepare CPU input colors (working-space scene-referred linear). ---
	// Indices are referenced by name below for targeted assertions.
	enum
	{
		Idx_Black = 0,
		Idx_Gray002,
		Idx_Gray018,   // mid gray
		Idx_Gray05,
		Idx_Gray10,
		Idx_Gray40,
		Idx_Red05,
		Idx_Green05,
		Idx_Blue05,
		Idx_Red10,
		Idx_Green10,
		Idx_Blue10,
		Idx_Bright8,
		Idx_Negative,
		Idx_Mixed,
		Idx_Count
	};

	TArray<FVector4f> InputColors;
	InputColors.SetNumZeroed(Idx_Count);
	InputColors[Idx_Black]    = FVector4f(0.0f, 0.0f, 0.0f, 0.0f);
	InputColors[Idx_Gray002]  = FVector4f(0.02f, 0.02f, 0.02f, 0.0f);
	InputColors[Idx_Gray018]  = FVector4f(0.18f, 0.18f, 0.18f, 0.0f);
	InputColors[Idx_Gray05]   = FVector4f(0.5f, 0.5f, 0.5f, 0.0f);
	InputColors[Idx_Gray10]   = FVector4f(1.0f, 1.0f, 1.0f, 0.0f);
	InputColors[Idx_Gray40]   = FVector4f(4.0f, 4.0f, 4.0f, 0.0f);
	InputColors[Idx_Red05]    = FVector4f(0.5f, 0.0f, 0.0f, 0.0f);
	InputColors[Idx_Green05]  = FVector4f(0.0f, 0.5f, 0.0f, 0.0f);
	InputColors[Idx_Blue05]   = FVector4f(0.0f, 0.0f, 0.5f, 0.0f);
	InputColors[Idx_Red10]    = FVector4f(1.0f, 0.0f, 0.0f, 0.0f);
	InputColors[Idx_Green10]  = FVector4f(0.0f, 1.0f, 0.0f, 0.0f);
	InputColors[Idx_Blue10]   = FVector4f(0.0f, 0.0f, 1.0f, 0.0f);
	InputColors[Idx_Bright8]  = FVector4f(8.0f, 8.0f, 8.0f, 0.0f);
	InputColors[Idx_Negative] = FVector4f(-0.01f, -0.01f, -0.01f, 0.0f);
	InputColors[Idx_Mixed]    = FVector4f(0.3f, 0.6f, 0.1f, 0.0f);

	const int32 NumColors = InputColors.Num();
	const uint32 NumBytes = NumColors * sizeof(FVector4f);

	// --- Display colors for the display round-trip (display-referred linear nits, sRGB gamut, [0, peak]). ---
	TArray<FVector4f> DisplayColors;
	DisplayColors.Add(FVector4f(0.0f,   0.0f,   0.0f,   0.0f));   // black
	DisplayColors.Add(FVector4f(10.0f,  10.0f,  10.0f,  0.0f));   // dark gray
	DisplayColors.Add(FVector4f(50.0f,  50.0f,  50.0f,  0.0f));   // mid gray
	DisplayColors.Add(FVector4f(90.0f,  90.0f,  90.0f,  0.0f));   // light gray
	DisplayColors.Add(FVector4f(100.0f, 100.0f, 100.0f, 0.0f));   // white
	DisplayColors.Add(FVector4f(100.0f, 0.0f,   0.0f,   0.0f));   // red
	DisplayColors.Add(FVector4f(0.0f,   100.0f, 0.0f,   0.0f));   // green
	DisplayColors.Add(FVector4f(0.0f,   0.0f,   100.0f, 0.0f));   // blue
	DisplayColors.Add(FVector4f(0.0f,   100.0f, 100.0f, 0.0f));   // cyan
	DisplayColors.Add(FVector4f(100.0f, 0.0f,   100.0f, 0.0f));   // magenta
	DisplayColors.Add(FVector4f(100.0f, 100.0f, 0.0f,   0.0f));   // yellow
	DisplayColors.Add(FVector4f(20.0f,  50.0f,  80.0f,  0.0f));   // mix
	const int32 NumDisplayColors = DisplayColors.Num();
	const uint32 DisplayNumBytes = NumDisplayColors * sizeof(FVector4f);

	// --- Limiting (display) color space = working color space for the SDR path (see GetTransformResourcesSDR). ---
	const UE::Color::FColorSpace& LimitingColorSpace = UE::Color::FColorSpace::GetWorking();
	const FMatrix44d& LimitingRgbToXYZ = LimitingColorSpace.GetRgbToXYZ();
	const FMatrix44d& LimitingXYZToRgb = LimitingColorSpace.GetXYZToRgb();

	// --- Output storage + readbacks (declared on the test stack, captured by reference). ---
	TArray<FVector4f> ForwardResult;   ForwardResult.SetNumZeroed(NumColors);
	TArray<FVector4f> RoundTripResult; RoundTripResult.SetNumZeroed(NumColors);
	TArray<FVector4f> DisplayRoundTripResult; DisplayRoundTripResult.SetNumZeroed(NumDisplayColors);
	FRHIGPUBufferReadback ForwardReadback(TEXT("ACES2Test.Forward"));
	FRHIGPUBufferReadback RoundTripReadback(TEXT("ACES2Test.RoundTrip"));
	FRHIGPUBufferReadback DisplayRoundTripReadback(TEXT("ACES2Test.DisplayRoundTrip"));

	FEvent* Signal = FGenericPlatformProcess::GetSynchEventFromPool(true);

	ENQUEUE_RENDER_COMMAND(ACESOutputTransformTest)(
		[&](FRHICommandListImmediate& RHICmdList)
		{
			FRDGBuilder GraphBuilder(RHICmdList);

			// Generate the real ACES 2.0 tables (Reach / Gamut cusp / Upper-hull gamma).
			FRHIShaderResourceView* ReachTableSRV = nullptr;
			FRHIShaderResourceView* GamutTableSRV = nullptr;
			FRHIShaderResourceView* GammaTableSRV = nullptr;
			UE::Color::ACES::GetTransformResources(GraphBuilder, SDRPeakLuminance, LimitingColorSpace,
				ReachTableSRV, GamutTableSRV, GammaTableSRV);

			// Input buffer (created + uploaded in one call).
			FRDGBufferRef InputBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("ACES2Test.Input"), InputColors);

			// Output buffers.
			const FRDGBufferDesc OutputDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), NumColors);
			FRDGBufferRef ForwardBuffer = GraphBuilder.CreateBuffer(OutputDesc, TEXT("ACES2Test.OutForward"));
			FRDGBufferRef RoundTripBuffer = GraphBuilder.CreateBuffer(OutputDesc, TEXT("ACES2Test.OutRoundTrip"));

			FACESOutputTransformTestCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FACESOutputTransformTestCS::FParameters>();
			PassParameters->WorkingColorSpace = GDefaultWorkingColorSpaceUniformBuffer.GetUniformBufferRef();
			PassParameters->ACESReachTable = ReachTableSRV;
			PassParameters->ACESGamutTable = GamutTableSRV;
			PassParameters->ACESGammaTable = GammaTableSRV;
			// Transposed to match the WorkingColorSpace convention (see FDefaultWorkingColorSpaceUniformBuffer::Update).
			PassParameters->LimitingRgbToXYZ = UE::Color::Transpose<float>(LimitingRgbToXYZ);
			PassParameters->LimitingXYZToRgb = UE::Color::Transpose<float>(LimitingXYZToRgb);
			PassParameters->bScaleWhite = 0;   // sRGB working == sRGB output gamut -> white points match.
			PassParameters->OutputGamut = OutputGamut_sRGB_D65;
			PassParameters->PeakLuminance = SDRPeakLuminance;
			PassParameters->NumColors = static_cast<uint32>(NumColors);
			PassParameters->InputColors = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(InputBuffer));
			PassParameters->OutForward = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ForwardBuffer));
			PassParameters->OutRoundTrip = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(RoundTripBuffer));

			FACESOutputTransformTestCS::FPermutationDomain ScenePermutation;   // DISPLAY_SIDE_ROUNDTRIP = false
			TShaderMapRef<FACESOutputTransformTestCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel), ScenePermutation);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ACES2OutputTransformTest"), ComputeShader,
				PassParameters, FIntVector(FMath::DivideAndRoundUp(NumColors, ThreadGroupSize), 1, 1));

			AddEnqueueCopyPass(GraphBuilder, &ForwardReadback, ForwardBuffer, NumBytes);
			AddEnqueueCopyPass(GraphBuilder, &RoundTripReadback, RoundTripBuffer, NumBytes);

			// --- Display-side round-trip pass: display -> scene -> display (DISPLAY_SIDE_ROUNDTRIP permutation). ---
			FRDGBufferRef DisplayInputBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("ACES2Test.DisplayInput"), DisplayColors);
			const FRDGBufferDesc DisplayBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), NumDisplayColors);
			FRDGBufferRef DisplayInterBuffer = GraphBuilder.CreateBuffer(DisplayBufferDesc, TEXT("ACES2Test.DisplayInter"));
			FRDGBufferRef DisplayRoundTripBuffer = GraphBuilder.CreateBuffer(DisplayBufferDesc, TEXT("ACES2Test.DisplayRoundTrip"));

			// Reuse the shared inputs (tables, matrices, scalars); only the buffers and count differ.
			FACESOutputTransformTestCS::FParameters* DisplayParameters = GraphBuilder.AllocParameters<FACESOutputTransformTestCS::FParameters>();
			*DisplayParameters = *PassParameters;
			DisplayParameters->NumColors = static_cast<uint32>(NumDisplayColors);
			DisplayParameters->InputColors = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(DisplayInputBuffer));
			DisplayParameters->OutForward = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(DisplayInterBuffer));
			DisplayParameters->OutRoundTrip = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(DisplayRoundTripBuffer));

			FACESOutputTransformTestCS::FPermutationDomain DisplayPermutation;
			DisplayPermutation.Set<FACESOutputTransformTestCS::FDisplaySideRoundTrip>(true);
			TShaderMapRef<FACESOutputTransformTestCS> DisplayShader(GetGlobalShaderMap(GMaxRHIFeatureLevel), DisplayPermutation);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ACES2DisplayRoundTrip"), DisplayShader,
				DisplayParameters, FIntVector(FMath::DivideAndRoundUp(NumDisplayColors, ThreadGroupSize), 1, 1));

			AddEnqueueCopyPass(GraphBuilder, &DisplayRoundTripReadback, DisplayRoundTripBuffer, DisplayNumBytes);

			GraphBuilder.Execute();
			RHICmdList.SubmitAndBlockUntilGPUIdle();

			{
				const FVector4f* Ptr = static_cast<const FVector4f*>(ForwardReadback.Lock(NumBytes));
				FMemory::Memcpy(ForwardResult.GetData(), Ptr, NumBytes);
				ForwardReadback.Unlock();
			}
			{
				const FVector4f* Ptr = static_cast<const FVector4f*>(RoundTripReadback.Lock(NumBytes));
				FMemory::Memcpy(RoundTripResult.GetData(), Ptr, NumBytes);
				RoundTripReadback.Unlock();
			}
			{
				const FVector4f* Ptr = static_cast<const FVector4f*>(DisplayRoundTripReadback.Lock(DisplayNumBytes));
				FMemory::Memcpy(DisplayRoundTripResult.GetData(), Ptr, DisplayNumBytes);
				DisplayRoundTripReadback.Unlock();
			}

			Signal->Trigger();
		});

	Signal->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(Signal);

	// --- Validation -------------------------------------------------------------------------

	// Invariant 1: every forward output is finite and non-negative (transform clamps negatives to >= 0).
	for (int32 i = 0; i < NumColors; ++i)
	{
		const FVector4f& C = ForwardResult[i];
		const bool bFinite = FMath::IsFinite(C.X) && FMath::IsFinite(C.Y) && FMath::IsFinite(C.Z);
		UTEST_TRUE(TEXT("Forward output is finite"), bFinite);
		UTEST_TRUE(TEXT("Forward output is non-negative"),
			C.X >= -KINDA_SMALL_NUMBER && C.Y >= -KINDA_SMALL_NUMBER && C.Z >= -KINDA_SMALL_NUMBER);
	}

	// Invariant 2: black maps to black.
	UTEST_EQUAL_TOLERANCE(TEXT("Black->Black R"), ForwardResult[Idx_Black].X, 0.0f, 1e-2f);
	UTEST_EQUAL_TOLERANCE(TEXT("Black->Black G"), ForwardResult[Idx_Black].Y, 0.0f, 1e-2f);
	UTEST_EQUAL_TOLERANCE(TEXT("Black->Black B"), ForwardResult[Idx_Black].Z, 0.0f, 1e-2f);

	// Invariant 3: neutral (achromatic) input stays achromatic in the (matching-white-point) output.
	const int32 GrayIndices[] = { Idx_Black, Idx_Gray002, Idx_Gray018, Idx_Gray05, Idx_Gray10, Idx_Gray40 };
	for (int32 Gi : GrayIndices)
	{
		const FVector4f& C = ForwardResult[Gi];
		UTEST_EQUAL_TOLERANCE(TEXT("Neutral stays achromatic (R==G)"), C.X, C.Y, 1e-3f);
		UTEST_EQUAL_TOLERANCE(TEXT("Neutral stays achromatic (R==B)"), C.X, C.Z, 1e-3f);
	}

	// Invariant 4: luminance is monotonically non-decreasing along the neutral ramp.
	for (int32 k = 1; k < UE_ARRAY_COUNT(GrayIndices); ++k)
	{
		const float Prev = ForwardResult[GrayIndices[k - 1]].X;
		const float Curr = ForwardResult[GrayIndices[k]].X;
		UTEST_TRUE(TEXT("Neutral ramp luminance is monotonic"), Curr >= Prev - 1e-3f);
	}

	// Invariant 5: forward output is bounded by the peak luminance.
	for (int32 i = 0; i < NumColors; ++i)
	{
		const FVector4f& C = ForwardResult[i];
		UTEST_TRUE(TEXT("Forward output is bounded by peak luminance"),
			C.X <= SDRPeakLuminance + 1e-1f && C.Y <= SDRPeakLuminance + 1e-1f && C.Z <= SDRPeakLuminance + 1e-1f);
	}

	/**
	 * Golden values: frozen regression baseline for the ACES 2.0 forward transform (display nits),
	 * captured from a known-good run of the current ACES 2.0 math (see ACESUtils.cpp). These are
	 * meant to detect math regressions (which shift outputs by whole nits), NOT to pin bit-exact GPU
	 * results -- the tolerance is loose enough to tolerate cross-vendor float differences. If a
	 * legitimate ACES change lands, re-capture from the "[ACES2 golden]" log lines below.
	 */
	struct FGolden { int32 Index; FVector3f Expected; };
	const float GoldenTolerance = 5e-2f;
	static const FGolden Goldens[] =
	{
		{ Idx_Gray018, FVector3f(9.999944f,  9.999951f,  9.999948f) },
		{ Idx_Red10,   FVector3f(55.405067f, 0.881294f,  0.329701f) },
		{ Idx_Green10, FVector3f(4.721551f,  51.437496f, 3.392059f) },
		{ Idx_Blue10,  FVector3f(0.0f,       0.0f,       51.341431f) },
	};
	for (const FGolden& G : Goldens)
	{
		const FVector4f& C = ForwardResult[G.Index];
		AddInfo(FString::Printf(TEXT("[ACES2 golden] Forward[%d] = (%.6f, %.6f, %.6f)"), G.Index, C.X, C.Y, C.Z));
		UTEST_EQUAL_TOLERANCE(TEXT("Golden forward R"), C.X, G.Expected.X, GoldenTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Golden forward G"), C.Y, G.Expected.Y, GoldenTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Golden forward B"), C.Z, G.Expected.Z, GoldenTolerance);
	}

	// Scene round-trip (scene -> forward -> display -> inverse -> scene) recovers the input tightly
	// for all in-range colors.
	const int32 RoundTripIndices[] = { Idx_Gray002, Idx_Gray018, Idx_Gray05, Idx_Mixed, Idx_Red05, Idx_Green05 };
	const float RoundTripTolerance = 5e-3f;
	for (int32 Ri : RoundTripIndices)
	{
		const FVector4f& In = InputColors[Ri];
		const FVector4f& Out = RoundTripResult[Ri];
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip R"), Out.X, In.X, RoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip G"), Out.Y, In.Y, RoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip B"), Out.Z, In.Z, RoundTripTolerance);
	}

	/**
	 * Scene round-trip, blue sRGB primary: its B channel only recovers to ~0.009, not RoundTripTolerance.
	 * This appears to be an intrinsic non-invertibility of the ACES 2.0 output transform near the blue
	 * primary rather than a bug in our implementation: the OpenColorIO 2.6 ACES 2.0 reference shows the
	 * same ~0.0089 miss on a verified scene->SDR->scene round-trip. R and G still recover tightly, so
	 * only B is widened. (The display round-trip below does not hit this -- see its note.)
	 */
	const float BlueChannelTolerance = 1.5e-2f;
	{
		const FVector4f& In = InputColors[Idx_Blue05];
		const FVector4f& Out = RoundTripResult[Idx_Blue05];
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip[Blue05] R"), Out.X, In.X, RoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip[Blue05] G"), Out.Y, In.Y, RoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Scene round-trip[Blue05] B"), Out.Z, In.Z, BlueChannelTolerance);
	}

	/**
	 * Display round-trip (display -> scene -> display) recovers every display color tightly,
	 * saturated primaries included -- no blue exception needed. Displayable colors are in the image
	 * of the forward transform, so inverting then re-forwarding returns to them (the OpenColorIO 2.6
	 * ACES 2.0 reference round-trips these to < 1e-4 normalized).
	 */
	const float DisplayRoundTripTolerance = 5e-2f;   // nits
	float MaxDisplayRoundTripError = 0.0f;
	for (int32 i = 0; i < NumDisplayColors; ++i)
	{
		const FVector4f& In = DisplayColors[i];
		const FVector4f& Out = DisplayRoundTripResult[i];
		MaxDisplayRoundTripError = FMath::Max(MaxDisplayRoundTripError,
			FMath::Max3(FMath::Abs(Out.X - In.X), FMath::Abs(Out.Y - In.Y), FMath::Abs(Out.Z - In.Z)));
		UTEST_EQUAL_TOLERANCE(TEXT("Display round-trip R"), Out.X, In.X, DisplayRoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Display round-trip G"), Out.Y, In.Y, DisplayRoundTripTolerance);
		UTEST_EQUAL_TOLERANCE(TEXT("Display round-trip B"), Out.Z, In.Z, DisplayRoundTripTolerance);
	}
	AddInfo(FString::Printf(TEXT("[ACES2 display round-trip] max error = %.5f nits"), MaxDisplayRoundTripError));

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
