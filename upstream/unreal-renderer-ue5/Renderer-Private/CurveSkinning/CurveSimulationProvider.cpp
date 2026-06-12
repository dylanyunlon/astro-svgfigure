// Copyright Epic Games, Inc. All Rights Reserved.

#include "CurveSimulationProvider.h"
#include "CurveSkinningSceneExtensionProxy.h"
#include "RenderGraphUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "ShaderCompilerCore.h"

static float GCurveSimulationWaveAmplitude = 5.0f;
static FAutoConsoleVariableRef CVarCurveSimulationWaveAmplitude(
	TEXT("r.CurveSkinning.Simulation.WaveAmplitude"),
	GCurveSimulationWaveAmplitude,
	TEXT("Amplitude (cm) of the debug wave-motion applied to simulated curves (EXPERIMENTAL: for debugging/prototyping only)."),
	ECVF_RenderThreadSafe
);

static float GCurveSimulationWaveFrequency = 1.5f;
static FAutoConsoleVariableRef CVarCurveSimulationWaveFrequency(
	TEXT("r.CurveSkinning.Simulation.WaveFrequency"),
	GCurveSimulationWaveFrequency,
	TEXT("Spatial frequency of the debug wave-motion applied to simulated curves (EXPERIMENTAL: for debugging/prototyping only)."),
	ECVF_RenderThreadSafe
);

///////////////////////////////////////////////////////////////////////////////////////////////////
// FGlobalCurveSolverCS
// Dummy solver, producing wave movement

class FGlobalCurveSolverCS : public FGlobalShader
{
public:
	static constexpr uint32 PointsPerGroup = 64u;

	DECLARE_GLOBAL_SHADER(FGlobalCurveSolverCS);
	SHADER_USE_PARAMETER_STRUCT(FGlobalCurveSolverCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer,   Headers)
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer,   RestPositionBuffer)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutDeformedPositionBuffer)
		SHADER_PARAMETER(uint32, HeaderIndex)
		SHADER_PARAMETER(float,  CurrentTime)
		SHADER_PARAMETER(float,  WaveAmplitude)
		SHADER_PARAMETER(float,  WaveFrequency)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return true;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_WarningsAsErrors);
		OutEnvironment.CompilerFlags.Add(CFLAG_HLSL2021);
		OutEnvironment.SetDefine(TEXT("POINTS_PER_GROUP"), PointsPerGroup);
	}
};

IMPLEMENT_GLOBAL_SHADER(FGlobalCurveSolverCS, "/Engine/Private/CurveSkinning/CurveSimulationProvider.usf", "GlobalCurveSolverCS", SF_Compute);

///////////////////////////////////////////////////////////////////////////////////////////////////

void ProvideCurveSimulationData(FCurveSkinningDataProvider::FProviderContext& Context)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(ProvideCurveSimulationData);

	FRDGBuilder& GraphBuilder = Context.GraphBuilder;
	const float  CurrentTime  = (float)Context.CurrentTime.GetRealTimeSeconds();

	for (const FCurveSkinningDataProvider::FProviderIndirection& Indirection : Context.Indirections)
	{
		// CPU-side early-skip — avoid issuing a no-op dispatch for primitives whose
		// current-slot data is not marked dirty this frame.
		if (!EnumHasAnyFlags(Indirection.DirtyCurveData, EDirtyCurveData::Current))
		{
			continue;
		}

		if (Indirection.NumCurves == 0)
		{
			continue;
		}

		FGlobalCurveSolverCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FGlobalCurveSolverCS::FParameters>();
		PassParameters->Headers                   = GraphBuilder.CreateSRV(Context.HeaderBuffer);
		PassParameters->RestPositionBuffer        = GraphBuilder.CreateSRV(Context.RestBuffer);
		PassParameters->OutDeformedPositionBuffer = GraphBuilder.CreateUAV(Context.DeformedBuffer);
		PassParameters->HeaderIndex               = Indirection.HeaderIndex;
		PassParameters->CurrentTime               = CurrentTime;
		PassParameters->WaveAmplitude             = GCurveSimulationWaveAmplitude;
		PassParameters->WaveFrequency             = GCurveSimulationWaveFrequency;

		auto ComputeShader = GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FGlobalCurveSolverCS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("GlobalCurveSolver(HeaderIndex=%u Curves=%u)", Indirection.HeaderIndex, Indirection.NumCurves),
			ComputeShader,
			PassParameters,
			FIntVector(Indirection.NumCurves, 1, 1)
		);
	}
}
