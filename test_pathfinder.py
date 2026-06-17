import sys
import os

# Add root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.utils.pathfinder import ParkingPathfinder

def test_dijkstra():
    config_path = "backend/marked_slots/parking_slots.json"
    if not os.path.exists(config_path):
        print(f"Graph config path not found: {config_path}")
        return

    pathfinder = ParkingPathfinder(config_path)
    
    # 1. Test Dijkstra between entry and slot 1
    path, cost = pathfinder.dijkstra("entry", "1")
    print(f"Path to slot 1: {path} (Cost: {cost:.2f})")
    
    # 2. Test Dijkstra between entry and slot 3
    path3, cost3 = pathfinder.dijkstra("entry", "3")
    print(f"Path to slot 3: {path3} (Cost: {cost3:.2f})")
    
    # 3. Test finding closest path
    res = pathfinder.find_shortest_path_to_available_slot(["1", "2", "3"])
    print(f"Shortest path to available slots [1, 2, 3]: {res}")

if __name__ == "__main__":
    test_dijkstra()
