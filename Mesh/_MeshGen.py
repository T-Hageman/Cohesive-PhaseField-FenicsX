from typing import Any, Optional, Tuple

import numpy as np
import gmsh
from shapely.geometry import Polygon, MultiPolygon, Point
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

from Utils.mpi_utils import mprint, comm, rank, size


class _MeshGen:
    """
    Mix-in containing mesh generation utilities leveraging gmsh.
    """

    def __init__(self, params: Any) -> None:
        if params.MeshType == "Rectangle":
            self._generate_rectangle_mesh(params)
        elif params.MeshType == "PlateWithHole":
            self._generate_plate_with_hole_mesh(params)
        elif params.MeshType == "FullPlateWithHole":
            self._generate_full_plate_with_hole_mesh(params)
        elif params.MeshType == "SingleEdgeNotched":
            self._generate_single_edge_notched_mesh(params)
        elif params.MeshType == "DamageChallenge":
            self._generate_damage_challenge_mesh(params)
        elif params.MeshType == "Cylinder":
            self._generate_cylinder_mesh(params)
        else:
            raise NotImplementedError(f"Mesh type {params.MeshType} not implemented.")

        super().__init__(params)
        
    def _generate_rectangle_mesh(self, params: Any) -> None:
        if (params.dim == 2):
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("rectangle")

            # Define rectangle corners
            p1 = gmsh.model.geo.addPoint(0, 0, 0)
            p2 = gmsh.model.geo.addPoint(params.Lx, 0, 0)
            p3 = gmsh.model.geo.addPoint(params.Lx, params.Ly, 0)
            p4 = gmsh.model.geo.addPoint(0, params.Ly, 0)

            # Define lines
            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)

            # Define curve loop and surface
            cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surface = gmsh.model.geo.addPlaneSurface([cl])

            # Synchronize geometry
            gmsh.model.geo.synchronize()
            
            # Add physical groups (required by dolfinx)
            gmsh.model.addPhysicalGroup(2, [surface], 1)  # 2D surface
            gmsh.model.setPhysicalName(2, 1, "Domain")
            
            # Add physical groups for boundaries
            gmsh.model.addPhysicalGroup(1, [l1], 1)  # Bottom
            gmsh.model.setPhysicalName(1, 1, "Bottom")
            gmsh.model.addPhysicalGroup(1, [l2], 2)  # Right
            gmsh.model.setPhysicalName(1, 2, "Right")
            gmsh.model.addPhysicalGroup(1, [l3], 3)  # Top
            gmsh.model.setPhysicalName(1, 3, "Top")
            gmsh.model.addPhysicalGroup(1, [l4], 4)  # Left
            gmsh.model.setPhysicalName(1, 4, "Left")
            
            # Set mesh size and generate
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.dx)
            gmsh.model.mesh.generate(2)

            # Convert to dolfinx mesh
            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()
            
            # Add named markers for boundaries and domain
            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)

        elif params.dim == 3:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("box")

            # Create rectangular prism volume
            volume = gmsh.model.occ.addBox(0, 0, 0, params.Lx, params.Ly, params.Lz)
            gmsh.model.occ.synchronize()

            # Add physical volume group
            gmsh.model.addPhysicalGroup(3, [volume], 1)
            gmsh.model.setPhysicalName(3, 1, "Domain")

            # Identify boundary surfaces by their bounding boxes
            boundary_groups = {
                "left": [],
                "right": [],
                "front": [],
                "back": [],
                "bottom": [],
                "top": []
            }
            max_dim = max(params.Lx, params.Ly, params.Lz)
            tol = max(1e-6 * max_dim, 1e-9)
            for _, tag in gmsh.model.occ.getEntities(2):
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
                if np.isclose(xmin, xmax, atol=tol):
                    if np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif np.isclose(xmin, params.Lx, atol=tol):
                        boundary_groups["right"].append(tag)
                elif np.isclose(ymin, ymax, atol=tol):
                    if np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["front"].append(tag)
                    elif np.isclose(ymin, params.Ly, atol=tol):
                        boundary_groups["back"].append(tag)
                elif np.isclose(zmin, zmax, atol=tol):
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, params.Lz, atol=tol):
                        boundary_groups["top"].append(tag)

            surface_markers = {
                "left": (1, "Left"),
                "right": (2, "Right"),
                "front": (3, "Front"),
                "back": (4, "Back"),
                "bottom": (5, "Bottom"),
                "top": (6, "Top")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                gmsh.model.addPhysicalGroup(2, surfaces, marker)
                gmsh.model.setPhysicalName(2, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.dx)
            gmsh.model.mesh.generate(3)

            # Convert to dolfinx mesh
            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("left", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("front", 3)
            self.add_boundary_marker("back", 4)
            self.add_boundary_marker("bottom", 5)
            self.add_boundary_marker("top", 6)

        else:
            raise NotImplementedError("Only 2D and 3D rectangle mesh generation is implemented.")

    def _generate_plate_with_hole_mesh(self, params: Any) -> None:
        if params.dim == 2:
            LX = params.Lx
            LY = params.Ly
            R = params.R
            dx = params.dx

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("plate_with_hole")

            p0 = gmsh.model.geo.addPoint(0, 0, 0)
            p1 = gmsh.model.geo.addPoint(R, 0, 0)
            p2 = gmsh.model.geo.addPoint(LX, 0, 0)
            p3 = gmsh.model.geo.addPoint(LX, LY, 0)
            p4 = gmsh.model.geo.addPoint(0, LY, 0)
            p5 = gmsh.model.geo.addPoint(0, R, 0)

            l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
            l2 = gmsh.model.geo.addLine(p2, p3)  # Right
            l3 = gmsh.model.geo.addLine(p3, p4)  # Top
            l4 = gmsh.model.geo.addLine(p4, p5)  # Left
            l5 = gmsh.model.geo.addCircleArc(p5, p0, p1)  # Hole

            cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5])
            surface = gmsh.model.geo.addPlaneSurface([cl])

            gmsh.model.geo.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface], 1)
            gmsh.model.setPhysicalName(2, 1, "Domain")

            gmsh.model.addPhysicalGroup(1, [l1], 1)
            gmsh.model.setPhysicalName(1, 1, "Bottom")
            gmsh.model.addPhysicalGroup(1, [l2], 2)
            gmsh.model.setPhysicalName(1, 2, "Right")
            gmsh.model.addPhysicalGroup(1, [l3], 3)
            gmsh.model.setPhysicalName(1, 3, "Top")
            gmsh.model.addPhysicalGroup(1, [l4], 4)
            gmsh.model.setPhysicalName(1, 4, "Left")
            gmsh.model.addPhysicalGroup(1, [l5], 5)
            gmsh.model.setPhysicalName(1, 5, "Hole")

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(2)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self._apply_damage_challenge_facet_tags(params)

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("hole", 5)

        elif params.dim == 3:
            LX = params.Lx
            LY = params.Ly
            LZ = params.Lz
            R = params.R
            dx = params.dx

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("plate_with_hole_3d")

            box = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
            cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, LZ, R)
            cut = gmsh.model.occ.cut([(3, box)], [(3, cylinder)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            volumes = [tag for dim, tag in cut[0] if dim == 3]
            if not volumes:
                raise RuntimeError("Plate-with-hole 3D mesh generation failed to create a volume.")

            gmsh.model.addPhysicalGroup(3, volumes, 1)
            gmsh.model.setPhysicalName(3, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "hole": [],
                "front": [],
                "back": []
            }
            max_dim = max(LX, LY, LZ, R)
            tol = max(1e-6 * max_dim, 1e-9)

            for vol in volumes:
                for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    z_planar = np.isclose(zmin, zmax, atol=tol)

                    if x_planar:
                        if np.isclose(xmin, 0.0, atol=tol):
                            boundary_groups["left"].append(tag)
                        elif np.isclose(xmin, LX, atol=tol):
                            boundary_groups["right"].append(tag)
                    elif y_planar:
                        if np.isclose(ymin, 0.0, atol=tol):
                            boundary_groups["bottom"].append(tag)
                        elif np.isclose(ymin, LY, atol=tol):
                            boundary_groups["top"].append(tag)
                    elif z_planar:
                        if np.isclose(zmin, 0.0, atol=tol):
                            boundary_groups["front"].append(tag)
                        elif np.isclose(zmin, LZ, atol=tol):
                            boundary_groups["back"].append(tag)
                    else:
                        boundary_groups["hole"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "hole": (5, "Hole"),
                "front": (6, "Front"),
                "back": (7, "Back")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                if surfaces:
                    gmsh.model.addPhysicalGroup(2, surfaces, marker)
                    gmsh.model.setPhysicalName(2, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(3)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self._apply_damage_challenge_facet_tags(params)

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("hole", 5)
            self.add_boundary_marker("front", 6)
            self.add_boundary_marker("back", 7)

        else:
            raise NotImplementedError("Only 2D and 3D plate-with-hole mesh generation is implemented.")

    def _generate_full_plate_with_hole_mesh(self, params: Any) -> None:
        if params.dim == 2:
            LX = params.Lx
            LY = params.Ly
            R = params.R
            dx = params.dx
            ellipticity = getattr(params, "ellipticity", 1.0)
            rotation_angle = getattr(params, "rotation_angle", 0.0)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("full_plate_with_hole")

            plate = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)
            # Create elliptical hole: Rx = R, Ry = R / ellipticity
            Rx = R
            Ry = R / ellipticity
            hole = gmsh.model.occ.addDisk(0.5 * LX, 0.5 * LY, 0, Rx, Ry)
            if not np.isclose(rotation_angle, 0.0):
                gmsh.model.occ.rotate(
                    [(2, hole)],
                    0.5 * LX, 0.5 * LY, 0,  # rotation center
                    0, 0, 1,                  # rotation axis (z)
                    np.radians(rotation_angle)
                )
            cut = gmsh.model.occ.cut([(2, plate)], [(2, hole)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            surfaces = [tag for dim, tag in cut[0] if dim == 2]
            if not surfaces:
                raise RuntimeError("Full plate-with-hole mesh generation failed to create a surface.")

            gmsh.model.addPhysicalGroup(2, surfaces, 1)
            gmsh.model.setPhysicalName(2, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "hole": []
            }
            max_dim = max(LX, LY, R)
            tol = max(1e-6 * max_dim, 1e-9)

            for surf in surfaces:
                for _, tag in gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    if y_planar and np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif y_planar and np.isclose(ymin, LY, atol=tol):
                        boundary_groups["top"].append(tag)
                    elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif x_planar and np.isclose(xmin, LX, atol=tol):
                        boundary_groups["right"].append(tag)
                    else:
                        boundary_groups["hole"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "hole": (5, "Hole")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                if surfaces:
                    gmsh.model.addPhysicalGroup(1, surfaces, marker)
                    gmsh.model.setPhysicalName(1, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(2)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("hole", 5)

        elif params.dim == 3:
            LX = params.Lx
            LY = params.Ly
            LZ = params.Lz
            R = params.R
            dx = params.dx
            ellipticity = getattr(params, "ellipticity", 1.0)
            rotation_angle = getattr(params, "rotation_angle", 0.0)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("full_plate_with_hole_3d")

            box = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
            # Create elliptical disk and extrude to form cylinder
            Rx = R
            Ry = R / ellipticity
            disk = gmsh.model.occ.addDisk(0.5 * LX, 0.5 * LY, 0, Rx, Ry)
            if not np.isclose(rotation_angle, 0.0):
                gmsh.model.occ.rotate(
                    [(2, disk)],
                    0.5 * LX, 0.5 * LY, 0,
                    0, 0, 1,
                    np.radians(rotation_angle)
                )
            extrusion = gmsh.model.occ.extrude([(2, disk)], 0, 0, LZ)
            tool_volumes = [(dim, tag) for dim, tag in extrusion if dim == 3]
            cut = gmsh.model.occ.cut([(3, box)], tool_volumes, removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            volumes = [tag for dim, tag in cut[0] if dim == 3]
            if not volumes:
                raise RuntimeError("Full plate-with-hole 3D mesh generation failed to create a volume.")

            gmsh.model.addPhysicalGroup(3, volumes, 1)
            gmsh.model.setPhysicalName(3, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "hole": [],
                "front": [],
                "back": []
            }
            max_dim = max(LX, LY, LZ, R)
            tol = max(1e-6 * max_dim, 1e-9)

            for vol in volumes:
                for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    z_planar = np.isclose(zmin, zmax, atol=tol)

                    if x_planar:
                        if np.isclose(xmin, 0.0, atol=tol):
                            boundary_groups["left"].append(tag)
                        elif np.isclose(xmin, LX, atol=tol):
                            boundary_groups["right"].append(tag)
                    elif y_planar:
                        if np.isclose(ymin, 0.0, atol=tol):
                            boundary_groups["bottom"].append(tag)
                        elif np.isclose(ymin, LY, atol=tol):
                            boundary_groups["top"].append(tag)
                    elif z_planar:
                        if np.isclose(zmin, 0.0, atol=tol):
                            boundary_groups["front"].append(tag)
                        elif np.isclose(zmin, LZ, atol=tol):
                            boundary_groups["back"].append(tag)
                    else:
                        boundary_groups["hole"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "hole": (5, "Hole"),
                "front": (6, "Front"),
                "back": (7, "Back")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                if surfaces:
                    gmsh.model.addPhysicalGroup(2, surfaces, marker)
                    gmsh.model.setPhysicalName(2, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(3)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("hole", 5)
            self.add_boundary_marker("front", 6)
            self.add_boundary_marker("back", 7)

        else:
            raise NotImplementedError("Only 2D and 3D full plate-with-hole mesh generation is implemented.")

    def _generate_single_edge_notched_mesh(self, params: Any) -> None:
        if params.dim == 2:
            LX = params.Lx
            LY = params.Ly
            LNotch = params.LNotch
            dx = params.dx

            notch_height = getattr(params, "NotchHeight", dx)
            notch_height = max(notch_height, 1.0e-6 * LY)
            notch_y0 = 0.5 * (LY - notch_height)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("single_edge_notched")

            plate = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)
            notch = gmsh.model.occ.addRectangle(0, notch_y0, 0, LNotch, notch_height)
            cut = gmsh.model.occ.cut([(2, plate)], [(2, notch)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            surfaces = [tag for dim, tag in cut[0] if dim == 2]
            if not surfaces:
                raise RuntimeError("Single-edge-notched mesh generation failed to create a surface.")

            gmsh.model.addPhysicalGroup(2, surfaces, 1)
            gmsh.model.setPhysicalName(2, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "notch": []
            }
            max_dim = max(LX, LY, LNotch)
            tol = max(1e-6 * max_dim, 1e-9)

            for surf in surfaces:
                for _, tag in gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    if y_planar and np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif y_planar and np.isclose(ymin, LY, atol=tol):
                        boundary_groups["top"].append(tag)
                    elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif x_planar and np.isclose(xmin, LX, atol=tol):
                        boundary_groups["right"].append(tag)
                    else:
                        boundary_groups["notch"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "notch": (5, "Notch")
            }
            for name, (marker, physical_name) in surface_markers.items():
                segments = boundary_groups[name]
                if segments:
                    gmsh.model.addPhysicalGroup(1, segments, marker)
                    gmsh.model.setPhysicalName(1, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(2)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("notch", 5)

        elif params.dim == 3:
            LX = params.Lx
            LY = params.Ly
            LZ = params.Lz
            LNotch = params.LNotch
            dx = params.dx

            notch_height = getattr(params, "NotchHeight", dx)
            notch_height = max(notch_height, 1.0e-6 * LY)
            notch_y0 = 0.5 * (LY - notch_height)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("single_edge_notched_3d")

            plate = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
            notch = gmsh.model.occ.addBox(0, notch_y0, 0, LNotch, notch_height, LZ)
            cut = gmsh.model.occ.cut([(3, plate)], [(3, notch)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            volumes = [tag for dim, tag in cut[0] if dim == 3]
            if not volumes:
                raise RuntimeError("Single-edge-notched 3D mesh generation failed to create a volume.")

            gmsh.model.addPhysicalGroup(3, volumes, 1)
            gmsh.model.setPhysicalName(3, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "notch": [],
                "front": [],
                "back": []
            }
            max_dim = max(LX, LY, LZ, LNotch)
            tol = max(1e-6 * max_dim, 1e-9)

            for vol in volumes:
                for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    z_planar = np.isclose(zmin, zmax, atol=tol)

                    if x_planar:
                        if np.isclose(xmin, 0.0, atol=tol):
                            boundary_groups["left"].append(tag)
                        elif np.isclose(xmin, LX, atol=tol):
                            boundary_groups["right"].append(tag)
                    elif y_planar:
                        if np.isclose(ymin, 0.0, atol=tol):
                            boundary_groups["bottom"].append(tag)
                        elif np.isclose(ymin, LY, atol=tol):
                            boundary_groups["top"].append(tag)
                    elif z_planar:
                        if np.isclose(zmin, 0.0, atol=tol):
                            boundary_groups["front"].append(tag)
                        elif np.isclose(zmin, LZ, atol=tol):
                            boundary_groups["back"].append(tag)
                    else:
                        boundary_groups["notch"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "notch": (5, "Notch"),
                "front": (6, "Front"),
                "back": (7, "Back")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                if surfaces:
                    gmsh.model.addPhysicalGroup(2, surfaces, marker)
                    gmsh.model.setPhysicalName(2, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(3)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("notch", 5)
            self.add_boundary_marker("front", 6)
            self.add_boundary_marker("back", 7)

        else:
            raise NotImplementedError("Only 2D and 3D single-edge-notched mesh generation is implemented.")

    def _generate_damage_challenge_mesh(self, params: Any) -> None:
        if params.dim == 2:
            LX = params.Lx
            LY = params.Ly
            LNotch = params.Lx_notch
            notch_thickness = params.Notch_Thickness
            notch_height = params.Ly_notch
            Lx_BC = params.Lx_BC
            dx = params.dx

            load_width = getattr(params, "Load_Width", dx)
            bc_width = getattr(params, "BC_Width", dx)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("damage_challenge")
            notch_x0 = LNotch - 0.5 * notch_thickness
            notch_x1 = LNotch + 0.5 * notch_thickness

            load_half = 0.5 * max(load_width, dx)
            bc_half = 0.5 * max(bc_width, dx)

            point_coords = {}
            point_cache = {}

            def _add_point(x, y):
                key = (round(x, 12), round(y, 12))
                existing = point_cache.get(key)
                if existing is not None:
                    return existing
                tag = gmsh.model.geo.addPoint(x, y, 0.0, dx)
                point_coords[tag] = (x, y)
                point_cache[key] = tag
                return tag

            def _sorted_unique(vals):
                vals = sorted(set(vals))
                return vals

            bottom_left_seg = (Lx_BC - bc_half, Lx_BC + bc_half)
            bottom_right_seg = (LX - Lx_BC - bc_half, LX - Lx_BC + bc_half)
            top_load_seg = (0.5 * LX - load_half, 0.5 * LX + load_half)

            def _segment_in_range(seg, x0, x1):
                return seg[1] > x0 and seg[0] < x1

            if not _segment_in_range(bottom_left_seg, 0.0, notch_x0) and not _segment_in_range(
                bottom_left_seg, notch_x1, LX
            ):
                mprint("Warning: bottom_left segment overlaps notch; it may be skipped.")
            if not _segment_in_range(bottom_right_seg, 0.0, notch_x0) and not _segment_in_range(
                bottom_right_seg, notch_x1, LX
            ):
                mprint("Warning: bottom_right segment overlaps notch; it may be skipped.")

            bottom_left_x = []
            if notch_x0 > 0.0:
                bottom_left_x = [0.0, notch_x0]
                if _segment_in_range(bottom_left_seg, 0.0, notch_x0):
                    bottom_left_x.extend(
                        [max(0.0, bottom_left_seg[0]), min(notch_x0, bottom_left_seg[1])]
                    )
            bottom_right_x = []
            if notch_x1 < LX:
                bottom_right_x = [notch_x1, LX]
                if _segment_in_range(bottom_right_seg, notch_x1, LX):
                    bottom_right_x.extend(
                        [max(notch_x1, bottom_right_seg[0]), min(LX, bottom_right_seg[1])]
                    )

            top_x = [0.0, LX, top_load_seg[0], top_load_seg[1]]
            top_x = [x for x in top_x if 0.0 <= x <= LX]

            bottom_left_x = _sorted_unique([x for x in bottom_left_x if 0.0 <= x <= notch_x0])
            bottom_right_x = _sorted_unique([x for x in bottom_right_x if notch_x1 <= x <= LX])
            top_x = _sorted_unique(top_x)

            bottom_lines = []
            top_lines = []
            left_lines = []
            right_lines = []
            notch_lines = []
            bottom_left_line = None
            bottom_right_line = None
            top_load_line = None

            def _add_line(p1, p2, store, tag_name=None):
                line = gmsh.model.geo.addLine(p1, p2)
                store.append(line)
                return line

            # Bottom left segment
            prev = _add_point(0.0, 0.0)
            for x in bottom_left_x[1:]:
                p = _add_point(x, 0.0)
                line = _add_line(prev, p, bottom_lines)
                mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
                if abs(mid_x - Lx_BC) <= bc_half + 1.0e-12 and bottom_left_line is None:
                    bottom_left_line = line
                prev = p

            # Notch
            p_notch_bl = prev
            p_notch_up = _add_point(notch_x0, notch_height)
            notch_lines.append(gmsh.model.geo.addLine(p_notch_bl, p_notch_up))
            p_notch_top = _add_point(notch_x1, notch_height)
            notch_lines.append(gmsh.model.geo.addLine(p_notch_up, p_notch_top))
            p_notch_br = _add_point(notch_x1, 0.0)
            notch_lines.append(gmsh.model.geo.addLine(p_notch_top, p_notch_br))

            # Bottom right segment
            prev = p_notch_br
            for x in bottom_right_x[1:]:
                p = _add_point(x, 0.0)
                line = _add_line(prev, p, bottom_lines)
                mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
                if abs(mid_x - (LX - Lx_BC)) <= bc_half + 1.0e-12 and bottom_right_line is None:
                    bottom_right_line = line
                prev = p

            # Right edge
            p_right_top = _add_point(LX, LY)
            right_lines.append(gmsh.model.geo.addLine(prev, p_right_top))

            # Top segment (right to left)
            prev = p_right_top
            for x in sorted(top_x, reverse=True)[1:]:
                p = _add_point(x, LY)
                line = _add_line(prev, p, top_lines)
                mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
                if abs(mid_x - 0.5 * LX) <= load_half + 1.0e-12 and top_load_line is None:
                    top_load_line = line
                prev = p

            # Left edge
            p_left_bottom = _add_point(0.0, 0.0)
            left_lines.append(gmsh.model.geo.addLine(prev, p_left_bottom))

            cl = gmsh.model.geo.addCurveLoop(
                bottom_lines + notch_lines + right_lines + top_lines + left_lines
            )
            surface = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface], 1)
            gmsh.model.setPhysicalName(2, 1, "Domain")

            if bottom_left_line is not None:
                gmsh.model.addPhysicalGroup(1, [bottom_left_line], 7)
                gmsh.model.setPhysicalName(1, 7, "Bottom_Left")
                gmsh.model.geo.mesh.setTransfiniteCurve(bottom_left_line, 2)
            if bottom_right_line is not None:
                gmsh.model.addPhysicalGroup(1, [bottom_right_line], 8)
                gmsh.model.setPhysicalName(1, 8, "Bottom_Right")
                gmsh.model.geo.mesh.setTransfiniteCurve(bottom_right_line, 2)
            if top_load_line is not None:
                gmsh.model.addPhysicalGroup(1, [top_load_line], 6)
                gmsh.model.setPhysicalName(1, 6, "Top_Load")
                gmsh.model.geo.mesh.setTransfiniteCurve(top_load_line, 2)

            if bottom_lines:
                gmsh.model.addPhysicalGroup(1, bottom_lines, 1)
                gmsh.model.setPhysicalName(1, 1, "Bottom")
            if right_lines:
                gmsh.model.addPhysicalGroup(1, right_lines, 2)
                gmsh.model.setPhysicalName(1, 2, "Right")
            if top_lines:
                gmsh.model.addPhysicalGroup(1, top_lines, 3)
                gmsh.model.setPhysicalName(1, 3, "Top")
            if left_lines:
                gmsh.model.addPhysicalGroup(1, left_lines, 4)
                gmsh.model.setPhysicalName(1, 4, "Left")
            if notch_lines:
                gmsh.model.addPhysicalGroup(1, notch_lines, 5)
                gmsh.model.setPhysicalName(1, 5, "Notch")

            gmsh.model.mesh.generate(2)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            mprint("DamageChallenge: applying custom facet tags (2D).")
            self._apply_damage_challenge_facet_tags(params)

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("notch", 5)
            self.add_boundary_marker("top_load", 6)
            self.add_boundary_marker("bottom_left", 7)
            self.add_boundary_marker("bottom_right", 8)
            self._print_damage_challenge_boundary_counts()

        elif params.dim == 3:
            LX = params.Lx
            LY = params.Ly
            LZ = params.Lz
            LNotch = params.Lx_notch
            notch_thickness = params.Notch_Thickness
            notch_height = params.Ly_notch
            Lx_BC = params.Lx_BC
            dx = params.dx

            load_width = getattr(params, "Load_Width", dx)
            bc_width = getattr(params, "BC_Width", dx)

            if isinstance(LNotch, (list, tuple, np.ndarray)):
                if len(LNotch) != 2:
                    raise ValueError("Lx_notch must have 2 components for 3D diagonal notches.")
                notch_x0 = float(LNotch[0])
                notch_x1 = float(LNotch[1])
            else:
                notch_x0 = float(LNotch)
                notch_x1 = float(LNotch)

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("damage_challenge_3d")

            plate = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)

            notch = gmsh.model.occ.addBox(
                -0.5 * notch_thickness,
                0,
                -0.5 * LZ,
                notch_thickness,
                notch_height,
                2.0 * LZ,
            )
            if abs(notch_x1 - notch_x0) > 0.0:
                theta = np.arctan2((notch_x1 - notch_x0), LZ)
                gmsh.model.occ.rotate([(3, notch)], 0, 0, 0, 0, 1, 0, theta)
            gmsh.model.occ.translate([(3, notch)], notch_x0, 0, 0)

            cut = gmsh.model.occ.cut([(3, plate)], [(3, notch)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()

            volumes = [tag for dim, tag in cut[0] if dim == 3]
            if not volumes:
                raise RuntimeError("DamageChallenge 3D mesh generation failed to create a volume.")

            gmsh.model.addPhysicalGroup(3, volumes, 1)
            gmsh.model.setPhysicalName(3, 1, "Domain")

            boundary_groups = {
                "bottom": [],
                "right": [],
                "top": [],
                "left": [],
                "notch": [],
                "top_load": [],
                "bottom_left": [],
                "bottom_right": [],
                "front": [],
                "back": []
            }
            max_dim = max(LX, LY, LZ, notch_thickness, notch_height, load_width, bc_width)
            tol = max(1e-6 * max_dim, 1e-9)

            for vol in volumes:
                for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
                    x_planar = np.isclose(xmin, xmax, atol=tol)
                    y_planar = np.isclose(ymin, ymax, atol=tol)
                    z_planar = np.isclose(zmin, zmax, atol=tol)
                    x_center = 0.5 * (xmin + xmax)

                    if y_planar and np.isclose(ymin, 0.0, atol=tol):
                        if abs(x_center - Lx_BC) <= 0.5 * bc_width:
                            boundary_groups["bottom_left"].append(tag)
                        elif abs(x_center - (LX - Lx_BC)) <= 0.5 * bc_width:
                            boundary_groups["bottom_right"].append(tag)
                        else:
                            boundary_groups["bottom"].append(tag)
                    elif y_planar and np.isclose(ymin, LY, atol=tol):
                        if abs(x_center - 0.5 * LX) <= 0.5 * load_width:
                            boundary_groups["top_load"].append(tag)
                        else:
                            boundary_groups["top"].append(tag)
                    elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif x_planar and np.isclose(xmin, LX, atol=tol):
                        boundary_groups["right"].append(tag)
                    elif z_planar:
                        if np.isclose(zmin, 0.0, atol=tol):
                            boundary_groups["front"].append(tag)
                        elif np.isclose(zmin, LZ, atol=tol):
                            boundary_groups["back"].append(tag)
                    else:
                        boundary_groups["notch"].append(tag)

            surface_markers = {
                "bottom": (1, "Bottom"),
                "right": (2, "Right"),
                "top": (3, "Top"),
                "left": (4, "Left"),
                "notch": (5, "Notch"),
                "top_load": (6, "Top_Load"),
                "bottom_left": (7, "Bottom_Left"),
                "bottom_right": (8, "Bottom_Right"),
                "front": (9, "Front"),
                "back": (10, "Back")
            }
            for name, (marker, physical_name) in surface_markers.items():
                surfaces = boundary_groups[name]
                if surfaces:
                    gmsh.model.addPhysicalGroup(2, surfaces, marker)
                    gmsh.model.setPhysicalName(2, marker, physical_name)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
            gmsh.model.mesh.generate(3)

            result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
            self.mesh = result[0]
            if len(result) > 1:
                self.cell_tags = result[1]
            if len(result) > 2:
                self.facet_tags = result[2]

            gmsh.finalize()

            mprint("DamageChallenge: applying custom facet tags (3D).")
            self._apply_damage_challenge_facet_tags(params)

            self.add_domain_marker("domain", 1)
            self.add_boundary_marker("bottom", 1)
            self.add_boundary_marker("right", 2)
            self.add_boundary_marker("top", 3)
            self.add_boundary_marker("left", 4)
            self.add_boundary_marker("notch", 5)
            self.add_boundary_marker("top_load", 6)
            self.add_boundary_marker("bottom_left", 7)
            self.add_boundary_marker("bottom_right", 8)
            self.add_boundary_marker("front", 9)
            self.add_boundary_marker("back", 10)
            self._print_damage_challenge_boundary_counts()

        else:
            raise NotImplementedError("Only 2D and 3D DamageChallenge mesh generation is implemented.")

    def _generate_cylinder_mesh(self, params: Any) -> None:
        """Generate a 3-D cylinder mesh (along the z-axis) with an optional
        circumferential radial notch on the outer surface.

        Required params
        ---------------
        R       – cylinder radius
        Lz      – cylinder height (z-extent)
        dx      – target mesh element size
        dim     – must be 3

        Optional params (notch)
        -----------------------
        notch_r  – radial depth of the notch (default 0 → no notch)
        notch_h  – z-position of the notch centre (default Lz/2)
        notch_dh – thickness of the notch in z (default dx)
        """
        if params.dim != 3:
            raise NotImplementedError(
                "Cylinder mesh generation is only implemented for 3D."
            )

        R = params.R
        Lz = params.Lz
        dx = params.dx
        notch_r = getattr(params, "notch_r", 0.0)
        notch_h = getattr(params, "notch_h", 0.5 * Lz)
        notch_dh = getattr(params, "notch_dh", dx)
        has_notch = notch_r > 0.0

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cylinder")

        # Main cylinder centred on (0, 0), z ∈ [0, Lz]
        cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, Lz, R)

        if has_notch:
            notch_z0 = notch_h - 0.5 * notch_dh
            # Build an annular ring (outer_radius > R so the cut is clean)
            outer_ring = gmsh.model.occ.addCylinder(
                0, 0, notch_z0, 0, 0, notch_dh, R + 1.0e-3
            )
            inner_ring = gmsh.model.occ.addCylinder(
                0, 0, notch_z0, 0, 0, notch_dh, R - notch_r
            )
            ring_cut = gmsh.model.occ.cut(
                [(3, outer_ring)], [(3, inner_ring)],
                removeObject=True, removeTool=True,
            )
            ring_vols = [(d, t) for d, t in ring_cut[0] if d == 3]
            body_cut = gmsh.model.occ.cut(
                [(3, cyl)], ring_vols,
                removeObject=True, removeTool=True,
            )
            volumes = [t for d, t in body_cut[0] if d == 3]
        else:
            volumes = [cyl]

        gmsh.model.occ.synchronize()

        if not volumes:
            raise RuntimeError(
                "Cylinder mesh generation failed to create a volume."
            )

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        # ---- classify boundary surfaces ----
        boundary_groups = {
            "bottom": [],
            "top": [],
            "outer": [],
        }
        if has_notch:
            boundary_groups["notch"] = []

        max_dim = max(R, Lz)
        tol = max(1e-6 * max_dim, 1e-9)

        for vol in volumes:
            for _, tag in gmsh.model.getBoundary(
                [(3, vol)], oriented=False, recursive=False
            ):
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    gmsh.model.occ.getBoundingBox(2, tag)
                )
                z_planar = np.isclose(zmin, zmax, atol=tol)

                if z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, Lz, atol=tol):
                        boundary_groups["top"].append(tag)
                    else:
                        # Annular face of the notch
                        if has_notch:
                            boundary_groups["notch"].append(tag)
                else:
                    if has_notch:
                        # Outer surface at r=R has x-extent ≈ 2R;
                        # notch inner surface at r=R-notch_r is smaller.
                        x_extent = xmax - xmin
                        if np.isclose(x_extent, 2.0 * R, atol=tol):
                            boundary_groups["outer"].append(tag)
                        else:
                            boundary_groups["notch"].append(tag)
                    else:
                        boundary_groups["outer"].append(tag)

        surface_markers = {
            "bottom": (1, "Bottom"),
            "top":    (2, "Top"),
            "outer":  (3, "Outer"),
        }
        if has_notch:
            surface_markers["notch"] = (4, "Notch")

        for name, (marker, physical_name) in surface_markers.items():
            surfaces = boundary_groups[name]
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, marker)
                gmsh.model.setPhysicalName(2, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(3)

        # Convert to dolfinx mesh
        result = model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
        self.mesh = result[0]
        if len(result) > 1:
            self.cell_tags = result[1]
        if len(result) > 2:
            self.facet_tags = result[2]

        gmsh.finalize()

        self.add_domain_marker("domain", 1)
        self.add_boundary_marker("bottom", 1)
        self.add_boundary_marker("top", 2)
        self.add_boundary_marker("outer", 3)
        if has_notch:
            self.add_boundary_marker("notch", 4)

    def _apply_damage_challenge_facet_tags(self, params: Any) -> None:
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        LX = params.Lx
        LY = params.Ly
        dx = params.dx
        Lx_BC = params.Lx_BC
        LNotch = params.Lx_notch
        notch_thickness = params.Notch_Thickness
        notch_height = params.Ly_notch

        load_width = getattr(params, "Load_Width", dx)
        bc_width = getattr(params, "BC_Width", dx)

        if tdim == 3:
            LZ = params.Lz
            if isinstance(LNotch, (list, tuple, np.ndarray)):
                if len(LNotch) != 2:
                    raise ValueError("Lx_notch must have 2 components for 3D diagonal notches.")
                notch_x0 = float(LNotch[0])
                notch_x1 = float(LNotch[1])
            else:
                notch_x0 = float(LNotch)
                notch_x1 = float(LNotch)
        else:
            LZ = 0.0
            notch_x0 = float(LNotch) - 0.5 * notch_thickness
            notch_x1 = float(LNotch) + 0.5 * notch_thickness

        max_dim = max(LX, LY, LZ, notch_thickness, notch_height, load_width, bc_width)
        tol = max(1e-6 * max_dim, 1e-9)
        tol = max(tol, 0.25 * dx)

        boundary_facets = dmesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        midpoints = dmesh.compute_midpoints(mesh, fdim, boundary_facets)
        x = midpoints[:, 0]
        y = midpoints[:, 1]
        if tdim == 3:
            z = midpoints[:, 2]

        top_mask = np.isclose(y, LY, atol=tol)
        bottom_mask = np.isclose(y, 0.0, atol=tol)
        left_mask = np.isclose(x, 0.0, atol=tol)
        right_mask = np.isclose(x, LX, atol=tol)

        index_map = mesh.topology.index_map(fdim)
        if hasattr(index_map, "global_indices"):
            try:
                global_ids = index_map.global_indices(False)
            except TypeError:
                global_ids = index_map.global_indices()
        elif hasattr(index_map, "local_to_global"):
            size = index_map.size_local + index_map.num_ghosts
            local_ids = np.arange(size, dtype=np.int32)
            global_ids = index_map.local_to_global(local_ids)
        else:
            raise AttributeError("IndexMap has no global index accessor.")
        boundary_global = global_ids[boundary_facets]

        def _global_closest_facet(mask, target):
            idx = np.where(mask)[0]
            if idx.size == 0:
                local_best = None
            else:
                dist2 = (x[idx] - target[0]) ** 2 + (y[idx] - target[1]) ** 2
                if tdim == 3:
                    dist2 += (z[idx] - target[2]) ** 2
                min_idx = int(np.argmin(dist2))
                best_pos = idx[min_idx]
                local_best = (float(dist2[min_idx]), int(boundary_global[best_pos]))

            gathered = comm.gather(local_best, root=0)
            if rank == 0:
                candidates = [g for g in gathered if g is not None]
                chosen = min(candidates, key=lambda v: (v[0], v[1])) if candidates else None
            else:
                chosen = None
            chosen = comm.bcast(chosen, root=0)
            if chosen is None:
                return np.array([], dtype=np.int32)
            chosen_gid = chosen[1]
            match = np.where(boundary_global == chosen_gid)[0]
            if match.size == 0:
                return np.array([], dtype=np.int32)
            return np.array([boundary_facets[match[0]]], dtype=np.int32)

        target_z = 0.5 * LZ if tdim == 3 else 0.0
        if tdim == 3:
            load_half = 0.5 * max(dx, load_width) + 0.5 * dx
            bc_half = 0.5 * max(dx, bc_width) + 0.5 * dx
            facets_by_tag = {
                "top_load": boundary_facets[
                    top_mask & (np.abs(x - 0.5 * LX) <= load_half + tol)
                ],
                "bottom_left": boundary_facets[
                    bottom_mask & (np.abs(x - Lx_BC) <= bc_half + tol)
                ],
                "bottom_right": boundary_facets[
                    bottom_mask & (np.abs(x - (LX - Lx_BC)) <= bc_half + tol)
                ],
                "top": boundary_facets[top_mask],
                "bottom": boundary_facets[bottom_mask],
                "left": boundary_facets[left_mask],
                "right": boundary_facets[right_mask],
            }
        else:
            facets_by_tag = {
                "top_load": _global_closest_facet(top_mask, (0.5 * LX, LY, target_z)),
                "bottom_left": _global_closest_facet(bottom_mask, (Lx_BC, 0.0, target_z)),
                "bottom_right": _global_closest_facet(bottom_mask, (LX - Lx_BC, 0.0, target_z)),
                "top": boundary_facets[top_mask],
                "bottom": boundary_facets[bottom_mask],
                "left": boundary_facets[left_mask],
                "right": boundary_facets[right_mask],
            }

        if tdim == 3:
            front_mask = np.isclose(z, 0.0, atol=tol)
            back_mask = np.isclose(z, LZ, atol=tol)
            facets_by_tag["front"] = boundary_facets[front_mask]
            facets_by_tag["back"] = boundary_facets[back_mask]

            z_safe = np.where(np.abs(LZ) < tol, tol, LZ)
            x_center = notch_x0 + (notch_x1 - notch_x0) * (z / z_safe)
            x_dist = np.abs(x - x_center)
            in_y = (y >= -tol) & (y <= notch_height + tol)
            on_side = in_y & (np.abs(x_dist - 0.5 * notch_thickness) <= tol)
            on_top = np.isclose(y, notch_height, atol=tol) & (
                x_dist <= 0.5 * notch_thickness + tol
            )
            facets_by_tag["notch"] = boundary_facets[on_side | on_top]
        else:
            notch_x0 = float(LNotch) - 0.5 * notch_thickness
            notch_x1 = float(LNotch) + 0.5 * notch_thickness
            in_y = (y >= -tol) & (y <= notch_height + tol)
            on_side = in_y & (
                np.isclose(x, notch_x0, atol=tol) | np.isclose(x, notch_x1, atol=tol)
            )
            on_top = np.isclose(y, notch_height, atol=tol) & (
                (x >= notch_x0 - tol) & (x <= notch_x1 + tol)
            )
            facets_by_tag["notch"] = boundary_facets[on_side | on_top]

        tag_order = [
            ("notch", 5),
            ("top_load", 6),
            ("bottom_left", 7),
            ("bottom_right", 8),
            ("top", 3),
            ("bottom", 1),
            ("left", 4),
            ("right", 2),
        ]
        if tdim == 3:
            tag_order.extend([("front", 9), ("back", 10)])

        assigned = set()
        facet_list = []
        value_list = []
        diagnostics = []
        for name, tag in tag_order:
            facets = np.unique(facets_by_tag.get(name, np.array([], dtype=np.int32)))
            diagnostics.append((name, tag, int(facets.size)))
            if facets.size == 0:
                continue
            new_facets = np.array([f for f in facets if f not in assigned], dtype=np.int32)
            if new_facets.size == 0:
                continue
            assigned.update(new_facets.tolist())
            facet_list.append(new_facets)
            value_list.append(np.full(new_facets.size, tag, dtype=np.int32))

        if facet_list:
            facets = np.concatenate(facet_list)
            values = np.concatenate(value_list)
            order = np.argsort(facets)
            self.facet_tags = dmesh.meshtags(mesh, fdim, facets[order], values[order])

        mprint("DamageChallenge boundary diagnostics:")
        for name, tag, count in diagnostics:
            global_count = comm.allreduce(count, op=MPI.SUM)
            mprint(f"  {name} (tag {tag}): {count} local, {global_count} global facets")

    def _print_damage_challenge_boundary_counts(self) -> None:
        if not hasattr(self, "facet_tags") or self.facet_tags is None:
            mprint("DamageChallenge boundary counts skipped: no facet tags.")
            return
        counts = []
        for name, tag in self.boundary_markers.items():
            try:
                facets = self.facet_tags.find(tag)
                counts.append((name, tag, len(facets)))
            except Exception:
                counts.append((name, tag, 0))
        mprint("DamageChallenge boundary counts (mesh tags):")
        for name, tag, count in counts:
            global_count = comm.allreduce(count, op=MPI.SUM)
            mprint(f"  {name} (tag {tag}): {count} local, {global_count} global facets")
        
