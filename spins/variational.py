#!/usr/bin/env python3
"""
DMRG upper bound (variational) for simple 1D spin-1/2 chains with PBC Hamiltonian couplings.

Heisenberg (J1) definition used here:
    H = (1/4) * sum_{i=1}^N sum_{a=x,y,z} J1 * sigma_i^a sigma_{i+1}^a, with N+1 ≡ 1 (PBC)

J1-J2 Heisenberg adds next-nearest neighbors:
    + (1/4) * sum_{i=1}^N sum_{a=x,y,z} J2 * sigma_i^a sigma_{i+2}^a, with N+2 ≡ 2 (PBC)

Ising model used here:
    H = -J * sum_{i=1}^N sigma_i^z sigma_{i+1}^z  - h * sum_{i=1}^N sigma_i^x  - k * sum_{i=1}^N sigma_i^z,
    with N+1 ≡ 1 (PBC)

TeNPy uses spin operators S^a = sigma^a / 2, so
  (1/4) sigma_i^a sigma_{i+1}^a = S_i^a S_{i+1}^a
Thus choosing Jx=Jy=Jz=1.0 implements exactly the same Hamiltonian.
"""

import time

import numpy as np

from tenpy.models.spins import SpinModel

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


class HeisenbergJ1J2Model(SpinModel):
    """Heisenberg J1-J2 model using the same spin operators as SpinModel.

    We build the nearest-neighbor terms via SpinModel.init_terms and then add
    an additional isotropic J2 coupling on `lat.pairs['next_nearest_neighbors']`.
    """

    def init_terms(self, model_params):
        super().init_terms(model_params)

        J2 = model_params.get("J2", 0.0, "real_or_array")
        if np.all(np.asarray(J2) == 0):
            return

        for u1, u2, dx in self.lat.pairs["next_nearest_neighbors"]:
            self.add_coupling(J2 / 2.0, u1, "Sp", u2, "Sm", dx, plus_hc=True)
            self.add_coupling(J2, u1, "Sz", u2, "Sz", dx)


def build_heisenberg_pbc_model(N: int, J: float = 1.0, conserve: str = "Sz"):
    model_params = dict(
        L=N,
        S=0.5,
        Jx=J, Jy=J, Jz=J,
        bc_MPS="finite",
        bc_x="periodic",     # PBC in the Hamiltonian (adds the long bond N-1)
        conserve=conserve,
        lattice="Chain",     # explicit (safe across versions)
    )
    return SpinModel(model_params)


def build_heisenberg_j1j2_pbc_model(N: int, J1: float = 1.0, J2: float = 0.0, conserve: str = "Sz"):
    model_params = dict(
        L=N,
        S=0.5,
        Jx=J1, Jy=J1, Jz=J1,
        J2=J2,
        bc_MPS="finite",
        bc_x="periodic",
        conserve=conserve,
        lattice="Chain",
    )
    return HeisenbergJ1J2Model(model_params)


def build_ising_pbc_model(N: int, J: float, h: float, k: float, conserve: str | None = None):
    """
    Target Hamiltonian:
        H = -J Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i - k Σ σ^z_i

    TeNPy SpinModel is written in terms of spin operators S = σ/2:
        σ^a = 2 S^a

    So:
        -J σ^z σ^z -> (-4J) Sz Sz
        -h σ^x     -> -(2h) Sx
        -k σ^z     -> -(2k) Sz

    TeNPy conventions inside SpinModel:
        H = Σ Jz * Sz_i Sz_j  - Σ hx * Sx_i - Σ hz * Sz_i + ...
    hence we set: Jz=-4J, hx=2h, hz=2k.
    """

    model_params = dict(
        L=N,
        S=0.5,
        Jx=0.0, Jy=0.0, Jz=-4.0 * J,
        hx=2.0 * h,
        hz=2.0 * k,
        bc_MPS="finite",
        bc_x="periodic",
        conserve=conserve,
        lattice="Chain",
    )
    return SpinModel(model_params)


def initial_product_state(model, kind: str = "neel"):
    N = model.lat.N_sites
    if kind == "neel":
        # |up down up down ...> (good for AFM Heisenberg)
        product_state = ["up" if (i % 2 == 0) else "down" for i in range(N)]
    elif kind == "all_up":
        product_state = ["up"] * N
    else:
        raise ValueError(f"Unknown init state kind: {kind}")

    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)
    psi.canonical_form()
    return psi


def dmrg_upper_bound(model, psi, chi_max: int, svd_min: float, max_E_err: float,
                     mixer: bool, dchi: int, nsweeps: int, combine: bool):
    # Ramp up chi to help convergence; final chi reaches chi_max.
    chi_schedule = dmrg.chi_list(chi_max=chi_max, dchi=dchi, nsweeps=nsweeps)

    dmrg_params = dict(
        mixer=mixer,
        max_E_err=max_E_err,
        chi_list=chi_schedule,
        trunc_params=dict(
            svd_min=svd_min,
        ),
        combine=combine,
    )

    t0 = time.perf_counter()
    info = dmrg.run(psi, model, dmrg_params)
    elapsed_s = time.perf_counter() - t0
    E = info["E"]  # total energy (variational upper bound)
    return E, psi, info, elapsed_s