"""
backend/utils/pathfinder.py

Implements Dijkstra's shortest path algorithm.
Loads the navigation graph from parking_slots.json and calculates coordinate paths 
from the entrance to the closest available parking slot.
"""

import json
import heapq
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ParkingPathfinder:
    def __init__(self, graph_config_path: str):
        self.nodes: Dict[str, List[int]] = {}
        self.graph: Dict[str, List[str]] = {}
        self._load(graph_config_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            print(f"[Pathfinder Warning] Graph config not found at: {path}")
            return
            
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.nodes = data.get("nodes", {})
            self.graph = data.get("graph", {})
            print(f"[Pathfinder] Loaded graph with {len(self.nodes)} nodes from {path}")
        except Exception as e:
            print(f"[Pathfinder Error] Error loading graph config: {e}")

    def _calculate_distance(self, p1: List[int], p2: List[int]) -> float:
        """Euclidean distance between two coordinate points."""
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    def dijkstra(self, start: str, end: str) -> Tuple[List[str], float]:
        """Calculates shortest path using Dijkstra's algorithm."""
        if start not in self.nodes or end not in self.nodes:
            return [], float("inf")

        queue = [(0.0, start, [])]
        visited = set()
        
        while queue:
            cost, node, path = heapq.heappop(queue)
            
            if node in visited:
                continue
                
            visited.add(node)
            path = path + [node]
            
            if node == end:
                return path, cost
                
            neighbors = self.graph.get(node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    dist = self._calculate_distance(self.nodes[node], self.nodes[neighbor])
                    heapq.heappush(queue, (cost + dist, neighbor, path))
                    
        return [], float("inf")

    def find_shortest_path_to_available_slot(self, available_slots: List[str]) -> Dict[str, any]:
        """Finds the shortest path from 'entry' to any available slot."""
        if not available_slots:
            return {"slot": "FULL", "coords": []}
            
        best_path = []
        best_cost = float("inf")
        best_slot = "FULL"
        
        for slot in available_slots:
            if slot not in self.nodes:
                continue
            path, cost = self.dijkstra("entry", slot)
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_slot = slot
                
        if best_cost == float("inf"):
            return {"slot": "FULL", "coords": []}
            
        coords = [self.nodes[node] for node in best_path]
        return {"slot": best_slot, "coords": coords}