"""Scene manager: cell registration, dirty tracking, frame setup.
Ported from upstream/unreal-renderer/SceneCore.cpp
"""

class AstroSceneManager:
    def __init__(self):
        self.cells = {}
        self.dirty = set()
    
    def register_cell(self, cell_id, bbox, species):
        self.cells[cell_id] = {"bbox": bbox, "species": species, "visible": True}
        self.dirty.add(cell_id)
    
    def mark_dirty(self, cell_id):
        self.dirty.add(cell_id)
    
    def begin_frame(self):
        frame_info = {"total_cells": len(self.cells), "dirty_cells": len(self.dirty), "dirty_list": list(self.dirty)}
        self.dirty.clear()
        return frame_info
    
    def get_visible_cells(self):
        return {cid: data for cid, data in self.cells.items() if data["visible"]}
