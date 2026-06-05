#!/usr/bin/env python3
"""Generate complete 5-station SUMO network.

Produces: five_station.nod.xml, five_station.edg.xml, five_station.rou.xml,
          five_station.sumocfg

Network layout (west to east):
  A ←1200m→ B ←2400m→ C ←1200m→ D ←1200m→ E

Each station: approach → throat_in → platforms → throat_out → depart
Platform counts: A=3, B=3, C=4, D=2, E=3
Platform length: 250m, Throat: ~80m, Approach/depart: ~100m
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT / "sumo" / "net"
ROUTES_DIR = PROJECT / "sumo" / "routes"
SUMO_DIR.mkdir(parents=True, exist_ok=True)
ROUTES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

PLATFORM_LEN = 250.0
THROAT_LEN = 80.0
APPROACH_LEN = 100.0
PLATFORM_SPACING = 40.0     # vertical spacing between platforms
THROAT_SPACING = 30.0        # vertical spacing for throat branches

CORRIDORS = {"A_B": 1200.0, "B_C": 2400.0, "C_D": 1200.0, "D_E": 1200.0}
PLATFORM_COUNTS = {"A": 3, "B": 3, "C": 4, "D": 2, "E": 3}
STATION_ORDER = ["A", "B", "C", "D", "E"]

# Station x-centers (cumulative corridor widths)
_station_x = 0.0
STATION_X: dict[str, float] = {}
for i, st in enumerate(STATION_ORDER):
    STATION_X[st] = _station_x
    if i < len(STATION_ORDER) - 1:
        key = f"{st}_{STATION_ORDER[i+1]}"
        _station_x += CORRIDORS.get(key, 1200.0)

# Station layout relative to center (west to east):
#   [spawn] --approach-- [throat_w] --throat_in--> [platform_w] ==platform== [platform_e] --throat_out--> [throat_e] --depart--> [exit]
# For up direction (west→east): spawn at west, exit at east
# For down direction (east→west): spawn at east, exit at west

# ---------------------------------------------------------------------------
# Node generation
# ---------------------------------------------------------------------------

def write_nodes():
    root = ET.Element("nodes")
    for st in STATION_ORDER:
        cx = STATION_X[st]
        n_plat = PLATFORM_COUNTS[st]

        # West side (up direction enters from west)
        ET.SubElement(root, "node", {
            "id": f"{st}_W_Spawn", "x": f"{cx - APPROACH_LEN - THROAT_LEN - 50}",
            "y": "0",
        })
        ET.SubElement(root, "node", {
            "id": f"{st}_Throat_W", "x": f"{cx - THROAT_LEN}",
            "y": "0",
        })
        # Platform west ends
        for p in range(n_plat):
            y = (p - (n_plat - 1) / 2.0) * PLATFORM_SPACING
            ET.SubElement(root, "node", {
                "id": f"{st}_p{p+1}_W", "x": f"{cx}",
                "y": f"{y}",
            })
        # Platform east ends
        for p in range(n_plat):
            y = (p - (n_plat - 1) / 2.0) * PLATFORM_SPACING
            ET.SubElement(root, "node", {
                "id": f"{st}_p{p+1}_E", "x": f"{cx + PLATFORM_LEN}",
                "y": f"{y}",
            })
        # East side
        ET.SubElement(root, "node", {
            "id": f"{st}_Throat_E", "x": f"{cx + PLATFORM_LEN + THROAT_LEN}",
            "y": "0",
        })
        ET.SubElement(root, "node", {
            "id": f"{st}_E_Exit", "x": f"{cx + PLATFORM_LEN + THROAT_LEN + APPROACH_LEN}",
            "y": "0",
        })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(SUMO_DIR / "five_station.nod.xml", encoding="UTF-8", xml_declaration=True)
    print(f"  Nodes → {SUMO_DIR / 'five_station.nod.xml'}")


# ---------------------------------------------------------------------------
# Edge generation
# ---------------------------------------------------------------------------

def write_edges():
    root = ET.Element("edges")

    for st_idx, st in enumerate(STATION_ORDER):
        n_plat = PLATFORM_COUNTS[st]
        cx = STATION_X[st]
        # Determine if station has east/west connections
        has_west = st_idx > 0
        has_east = st_idx < len(STATION_ORDER) - 1

        # --- UP direction (west → east) ---
        if has_west:
            # Approach from previous corridor
            prev = STATION_ORDER[st_idx - 1]
            key = f"{prev}_{st}"
            corr_len = CORRIDORS.get(key, 1200.0)
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_approach",
                "from": f"{prev}_Throat_E",
                "to": f"{st}_Throat_W",
                "speed": "30", "numLanes": "1", "allow": "rail",
                "length": f"{corr_len}",
            })
        else:
            # A station: spawn approach
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_approach",
                "from": f"{st}_W_Spawn", "to": f"{st}_Throat_W",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })

        # Throat in (common trunk then branches)
        throat_in_common = f"{st}_up_throat_in"
        ET.SubElement(root, "edge", {
            "id": throat_in_common,
            "from": f"{st}_Throat_W", "to": f"{st}_p1_W",
            "speed": "20", "numLanes": "1", "allow": "rail",
        })
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_in_{p+1}",
                "from": f"{st}_Throat_W", "to": f"{st}_p{p+1}_W",
                "speed": "20", "numLanes": "1", "allow": "rail",
            })

        # Platforms (west → east)
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_p{p+1}_up",
                "from": f"{st}_p{p+1}_W", "to": f"{st}_p{p+1}_E",
                "speed": "15", "numLanes": "1", "allow": "rail",
                "length": f"{PLATFORM_LEN}",
            })

        # Throat out
        ET.SubElement(root, "edge", {
            "id": f"{st}_up_throat_out",
            "from": f"{st}_p1_E", "to": f"{st}_Throat_E",
            "speed": "20", "numLanes": "1", "allow": "rail",
        })
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_out_{p+1}",
                "from": f"{st}_p{p+1}_E", "to": f"{st}_Throat_E",
                "speed": "20", "numLanes": "1", "allow": "rail",
            })

        if has_east:
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_depart",
                "from": f"{st}_Throat_E", "to": f"{STATION_ORDER[st_idx + 1]}_Throat_W",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })
        else:
            ET.SubElement(root, "edge", {
                "id": f"{st}_up_depart",
                "from": f"{st}_Throat_E", "to": f"{st}_E_Exit",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })

        # --- DOWN direction (east → west) ---
        if has_east:
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_approach",
                "from": f"{STATION_ORDER[st_idx + 1]}_Throat_W", "to": f"{st}_Throat_E",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })
        else:
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_approach",
                "from": f"{st}_E_Exit", "to": f"{st}_Throat_E",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })

        # Down throat in
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_in_{p+1}",
                "from": f"{st}_Throat_E", "to": f"{st}_p{p+1}_E",
                "speed": "20", "numLanes": "1", "allow": "rail",
            })

        # Platforms (east → west)
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_p{p+1}_down",
                "from": f"{st}_p{p+1}_E", "to": f"{st}_p{p+1}_W",
                "speed": "15", "numLanes": "1", "allow": "rail",
                "length": f"{PLATFORM_LEN}",
            })

        # Down throat out
        for p in range(n_plat):
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_out_{p+1}",
                "from": f"{st}_p{p+1}_W", "to": f"{st}_Throat_W",
                "speed": "20", "numLanes": "1", "allow": "rail",
            })

        if has_west:
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_depart",
                "from": f"{st}_Throat_W", "to": f"{STATION_ORDER[st_idx - 1]}_Throat_E",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })
        else:
            ET.SubElement(root, "edge", {
                "id": f"{st}_down_depart",
                "from": f"{st}_Throat_W", "to": f"{st}_W_Spawn",
                "speed": "30", "numLanes": "1", "allow": "rail",
            })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(SUMO_DIR / "five_station.edg.xml", encoding="UTF-8", xml_declaration=True)
    print(f"  Edges → {SUMO_DIR / 'five_station.edg.xml'}")


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------

def write_routes():
    root = ET.Element("routes")

    # vTypes
    vtypes = [
        ("G", 80, 83.3, "1,0,0"),
        ("D", 100, 69.4, "0,0,1"),
        ("Z", 140, 55.6, "0,1,0"),
        ("K", 160, 44.4, "1,0.5,0"),
    ]
    for vid, length, speed, color in vtypes:
        ET.SubElement(root, "vType", {
            "id": vid, "vClass": "rail", "length": str(length),
            "maxSpeed": str(speed), "accel": "1.0", "decel": "1.5",
            "sigma": "0.3", "minGap": "5", "color": color, "guiShape": "rail",
        })

    # Routes: up (A→E) and down (E→A)
    def make_route(rid, edges_list):
        ET.SubElement(root, "route", {"id": rid, "edges": " ".join(edges_list)})

    for p in range(1, 4):  # 3 platforms at A and E
        # Up route through platform p at each station
        make_route(f"up_p{p}", [
            "A_up_approach",
            f"A_up_in_{p}", f"A_p{p}_up", f"A_up_out_{p}",
            "A_up_depart",  # A→B corridor
            f"B_up_in_{p}", f"B_p{p}_up", f"B_up_out_{p}",
            "B_up_depart",  # B→C corridor
            f"C_up_in_{p}", f"C_p{p}_up", f"C_up_out_{p}",
            "C_up_depart",  # C→D corridor
            f"D_up_in_{min(p,2)}", f"D_p{min(p,2)}_up", f"D_up_out_{min(p,2)}",
            "D_up_depart",  # D→E corridor
            f"E_up_in_{p}", f"E_p{p}_up", f"E_up_out_{p}",
            "E_up_depart",
        ])
        if p <= 2:
            make_route(f"up_p{p}_d2", [
                "A_up_approach",
                f"A_up_in_{min(p,3)}", f"A_p{min(p,3)}_up", f"A_up_out_{min(p,3)}",
                "A_up_depart",
                f"B_up_in_{min(p,3)}", f"B_p{min(p,3)}_up", f"B_up_out_{min(p,3)}",
                "B_up_depart",
                f"C_up_in_{p}", f"C_p{p}_up", f"C_up_out_{p}",
                "C_up_depart",
                f"D_up_in_{p}", f"D_p{p}_up", f"D_up_out_{p}",
                "D_up_depart",
                f"E_up_in_{p}", f"E_p{p}_up", f"E_up_out_{p}",
                "E_up_depart",
            ])

        # Down route
        make_route(f"down_p{p}", [
            "E_down_approach",
            f"E_down_in_{p}", f"E_p{p}_down", f"E_down_out_{p}",
            "E_down_depart",  # E→D corridor
            f"D_down_in_{min(p,2)}", f"D_p{min(p,2)}_down", f"D_down_out_{min(p,2)}",
            "D_down_depart",  # D→C corridor
            f"C_down_in_{p}", f"C_p{p}_down", f"C_down_out_{p}",
            "C_down_depart",  # C→B corridor
            f"B_down_in_{p}", f"B_p{p}_down", f"B_down_out_{p}",
            "B_down_depart",  # B→A corridor
            f"A_down_in_{p}", f"A_p{p}_down", f"A_down_out_{p}",
            "A_down_depart",
        ])

    # Vehicles
    vehicles = [
        ("K641", "K", "up_p3", 0),
        ("G1408", "G", "down_p2", 90),
        ("D133", "D", "up_p1", 120),
        ("G5050", "G", "up_p2", 180),
        ("K999", "Z", "up_p2", 220),
        ("G5060", "G", "down_p3", 240),
        ("K667", "K", "up_p2", 300),
        ("Z386", "Z", "down_p1", 360),
        ("K1666", "K", "down_p2", 480),
    ]
    for vid, vtype, route_id, depart in vehicles:
        ET.SubElement(root, "vehicle", {
            "id": vid, "type": vtype, "route": route_id,
            "depart": str(depart), "departLane": "best", "departSpeed": "max",
        })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(ROUTES_DIR / "five_station.rou.xml", encoding="UTF-8", xml_declaration=True)
    print(f"  Routes → {ROUTES_DIR / 'five_station.rou.xml'}")


# ---------------------------------------------------------------------------
# SUMO config
# ---------------------------------------------------------------------------

def write_sumocfg():
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", {"value": str(SUMO_DIR / "five_station.net.xml")})
    ET.SubElement(inp, "route-files", {"value": str(ROUTES_DIR / "five_station.rou.xml")})

    time_e = ET.SubElement(root, "time")
    ET.SubElement(time_e, "begin", {"value": "0"})
    ET.SubElement(time_e, "end", {"value": "1200"})
    ET.SubElement(time_e, "step-length", {"value": "1.0"})

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", {"value": "600"})

    gui = ET.SubElement(root, "gui_only")
    ET.SubElement(gui, "delay", {"value": "50"})
    ET.SubElement(gui, "start", {"value": "true"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(PROJECT / "sumo" / "five_station.sumocfg", encoding="UTF-8", xml_declaration=True)
    print(f"  SUMO config → {PROJECT / 'sumo' / 'five_station.sumocfg'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Generating 5-station SUMO network...")
    write_nodes()
    write_edges()
    write_routes()
    write_sumocfg()
    print("\nDone. Next: run 'netconvert' to generate .net.xml")
    print(f"  cd {PROJECT}")
    print(f"  netconvert --node-files={SUMO_DIR / 'five_station.nod.xml'} --edge-files={SUMO_DIR / 'five_station.edg.xml'} --output-file={SUMO_DIR / 'five_station.net.xml'} --no-turnarounds=true")
    print(f"  sumo-gui -c sumo/five_station.sumocfg")


if __name__ == "__main__":
    main()
