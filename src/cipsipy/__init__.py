"""
CIPSIpy - Selected Configuration Interaction in JAX

A simple CIPSI implementation for learning JAX and quantum chemistry.
"""

import jax

# Enable high precision globally for all package modules
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
