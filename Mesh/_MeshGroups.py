from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from Utils.mpi_utils import mprint


class _MeshGroups:
    """
    Mix-in class for managing named mesh groups (boundaries and domains)
    """

    def __init__(self, params: Any) -> None:
        # Initialize boundary and domain name mappings
        self.boundary_markers: Dict[str, int] = {}
        self.domain_markers: Dict[str, int] = {}
        
        super().__init__(params)
    
    def add_boundary_marker(self, name: str, tag: int) -> None:
        """
        Add a named boundary marker.
        
        Args:
            name: Name of the boundary (e.g., "top", "bottom")
            tag: Integer tag associated with the boundary
        """
        self.boundary_markers[name] = tag
    
    def add_domain_marker(self, name: str, tag: int) -> None:
        """
        Add a named domain marker.
        
        Args:
            name: Name of the domain (e.g., "fluid", "solid")
            tag: Integer tag associated with the domain
        """
        self.domain_markers[name] = tag
    
    def get_boundary_tag(self, name: str) -> int:
        """
        Get the tag for a named boundary.
        
        Args:
            name: Name of the boundary
            
        Returns:
            Integer tag associated with the boundary
            
        Raises:
            KeyError: If boundary name not found
        """
        if name not in self.boundary_markers:
            raise KeyError(f"Boundary '{name}' not found. Available boundaries: {list(self.boundary_markers.keys())}")
        return self.boundary_markers[name]
    
    def get_domain_tag(self, name: str) -> int:
        """
        Get the tag for a named domain.
        
        Args:
            name: Name of the domain
            
        Returns:
            Integer tag associated with the domain
            
        Raises:
            KeyError: If domain name not found
        """
        if name not in self.domain_markers:
            raise KeyError(f"Domain '{name}' not found. Available domains: {list(self.domain_markers.keys())}")
        return self.domain_markers[name]
    
    def get_boundary_facets(self, name: str) -> np.ndarray:
        """
        Get facets for a named boundary.
        
        Args:
            name: Name of the boundary
            
        Returns:
            Array of facet indices for the boundary
        """
        tag = self.get_boundary_tag(name)
        return self.facet_tags.find(tag)

    def get_domain_cells(self, name: str) -> np.ndarray:
        """
        Get cells for a named domain.
        
        Args:
            name: Name of the domain
            
        Returns:
            Array of cell indices for the domain
        """
        tag = self.get_domain_tag(name)
        return self.cell_tags.find(tag)
    
    def list_boundaries(self) -> Dict[str, int]:
        """
        List all named boundaries and their tags.
        
        Returns:
            Dictionary mapping boundary names to tags
        """
        return self.boundary_markers.copy()
    
    def list_domains(self) -> Dict[str, int]:
        """
        List all named domains and their tags.
        
        Returns:
            Dictionary mapping domain names to tags
        """
        return self.domain_markers.copy()
    
    def print_mesh_groups(self) -> None:
        """
        Print all mesh groups (boundaries and domains) with their tags.
        """
        mprint("\n=== Mesh Groups ===")
        
        if self.domain_markers:
            mprint("\nDomains:")
            for name, tag in self.domain_markers.items():
                mprint(f"  {name}: tag={tag}")
        
        if self.boundary_markers:
            mprint("\nBoundaries:")
            for name, tag in self.boundary_markers.items():
                mprint(f"  {name}: tag={tag}")
        
        mprint("==================\n")
