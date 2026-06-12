// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "NaniteShared.h"
#include "NaniteCullRaster.h"
#include "SceneExtensions.h"
#include "Rendering/NaniteResources.h"
#include "Tasks/Task.h"

DECLARE_GPU_STAT_NAMED_EXTERN(NaniteEditor, TEXT("Nanite Editor"));

namespace Nanite
{

void DrawHitProxies(
	FRDGBuilder& GraphBuilder,
	const FScene& Scene,
	const FViewInfo& View,
	const FRasterResults& RasterResults,
	FRDGTextureRef HitProxyTexture,
	FRDGTextureRef HitProxyDeptTexture
);

FRDGBufferSRVRef GetEditorSelectedHitProxyIdsSRV(FRDGBuilder& GraphBuilder, TArrayView<const uint32> HitProxyIds);
FRDGBufferSRVRef GetEditorSelectedHitProxyIdsSRV(FRDGBuilder& GraphBuilder, const FScene& Scene);

#if WITH_EDITOR

void DrawEditorSelection(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef DepthTarget,
	FRDGTextureRef OverlayTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults* NaniteRasterResults,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect SceneDepthViewRect
);

void DrawEditorVisualizeLevelInstance(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef DepthTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults* NaniteRasterResults,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect SceneDepthViewRect
);

void EmitEditorNaniteSceneDepth(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef EditorDepthTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	const FRasterResults& NaniteRasterResults
);

/** Scene extension holding the Nanite editor draw lists for selection outline and level-instance visualization. */
class FEditorSceneExtension : public ISceneExtension
{
	DECLARE_SCENE_EXTENSION(RENDERER_API, FEditorSceneExtension);

public:
	using ISceneExtension::ISceneExtension;

	using FInstanceDrawList = TArray<Nanite::FInstanceDraw, SceneRenderingAllocator>;
	using FHitProxyIdList   = TArray<uint32, SceneRenderingAllocator>;

	struct FSelectionGroup
	{
		FInstanceDrawList DrawList;
		FHitProxyIdList   HitProxyIds;
	};

	enum class ESelectionGroup : uint32
	{
		/** Instances that are actually selected, either by component or individual selection. */
		Selected,
		/** Secondary selection type for views with selected components when the parent is selected but this component is not. */
		ParentOnlySelected,

		Num
	};

	class FUpdater : public ISceneExtensionUpdater
	{
		DECLARE_SCENE_EXTENSION_UPDATER(FUpdater, FEditorSceneExtension);

	public:
		explicit FUpdater(FEditorSceneExtension& InSceneData);
		virtual void End() override;
		virtual void PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet) override;
		virtual void PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet) override;

	private:
		FEditorSceneExtension* SceneData = nullptr;
		const bool bEnableAsync = true;
	};

	friend class FUpdater;

	static bool ShouldCreateExtension(FScene& InScene);

	virtual ISceneExtensionUpdater* CreateUpdater() override;

	/** Returns the selection group draw list for the given group. Syncs on the build task first. */
	const FSelectionGroup& GetSelectionGroup(ESelectionGroup Group) const;

	/** Returns the draw list used for the "visualize level instance" overlay. Syncs on the build task first. */
	const FInstanceDrawList& GetVisualizeLevelInstancesDrawList() const;

	/** Returns the draw list used to render overlay color in an independent pass */
	const FInstanceDrawList& GetOverlayDrawList() const;

	/** Syncs on the build task; call before reading any data directly. */
	void SyncOnDrawLists() const;

private:
	enum ETask : uint32
	{
		BuildDrawListsTask,

		NumTasks
	};

	TSet<FPersistentPrimitiveIndex> RelevantPrimitives;
	FInstanceDrawList VisualizeLevelInstancesDrawList;
	FInstanceDrawList OverlayDrawList;
	FSelectionGroup   SelectionGroups[uint32(ESelectionGroup::Num)];
	TStaticArray<UE::Tasks::FTask, NumTasks> TaskHandles;
	bool bEnabled = false;
};

#endif

} // namespace Nanite
