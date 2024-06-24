import jax_cosmo as jc
import jax.numpy as jnp
import numpy as np

ZMAX = 2.0
ELLMIN = 2
NELL = 30
ELLMAX_GC = 300
ELLMAX_WL = 3000


def get_nz(sfile, tracertype="wl"):
    tracers_names = list(sfile.tracers.keys())

    if tracertype == "wl":
        tname = "DESwl__"
    else:
        tname = "DESgc__"
    nbin = sum([tname in tracers_names[i] for i in range(len(tracers_names))])

    nz_distributions = list()
    for i in range(nbin):
        name = tname + str(i)
        distribution = sfile.tracers[name]
        jaxred = jc.redshift.custom_nz(
            distribution.z.astype("float64"),
            distribution.nz.astype("float64"),
            zmax=ZMAX,
        )
        nz_distributions.append(jaxred)

    return nz_distributions


def calculate_lmax_gc(sfile, kmax):
    tracers_names = list(sfile.tracers.keys())
    nbin_gc = sum(["DESgc__" in tracers_names[i] for i in range(len(tracers_names))])
    vanillacosmo = jc.Planck15()
    lmaxs = list()
    for i in range(nbin_gc):
        tracer = sfile.tracers[f"DESgc__{i}"]
        zmid = jnp.average(jnp.asarray(tracer.z), weights=jnp.asarray(tracer.nz))
        chi = jc.background.radial_comoving_distance(vanillacosmo, 1.0 / (1.0 + zmid))
        minmax = jnp.concatenate([10.0 * jnp.ones(1), kmax * chi - 0.5], dtype=int)
        lmax = jnp.max(minmax)
        lmaxs.append(lmax)
    return lmaxs


def scale_cuts(sfile, kmax=0.15, lmin_wl=30, lmax_wl=2000):
    # First we remove all B-modes
    sfile.remove_selection(data_type="cl_bb")
    sfile.remove_selection(data_type="cl_be")
    sfile.remove_selection(data_type="cl_eb")
    sfile.remove_selection(data_type="cl_0b")

    tracers_names = list(sfile.tracers.keys())
    nbin_gc = sum(["DESgc__" in tracers_names[i] for i in range(len(tracers_names))])
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])
    lmaxs_gc = calculate_lmax_gc(sfile, kmax)

    for i, lmax in enumerate(lmaxs_gc):
        print(f"Maximum ell is {lmax}")
        tname_1 = f"DESgc__{i}"

        # Remove from galaxy clustering
        sfile.remove_selection(
            data_type="cl_00", tracers=(tname_1, tname_1), ell__gt=lmax
        )

        # Remove from galaxy-galaxy lensing
        for j in range(nbin_wl):
            tname_2 = f"DESwl__{j}"
            sfile.remove_selection(
                data_type="cl_0e", tracers=(tname_1, tname_2), ell__gt=lmax
            )

    # apply scale cut for weak lensing
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tname_1 = f"DESwl__{i}"
            tname_2 = f"DESwl__{j}"
            sfile.remove_selection(
                data_type="cl_ee", tracers=(tname_1, tname_2), ell__gt=lmax_wl
            )
            sfile.remove_selection(
                data_type="cl_ee", tracers=(tname_1, tname_2), ell__lt=lmin_wl
            )

    return sfile


def get_data_type(tracer_combination):
    if "gc" in tracer_combination[0] and "gc" in tracer_combination[1]:
        dtype = "cl_00"
    elif "gc" in tracer_combination[0] and "wl" in tracer_combination[1]:
        dtype = "cl_0e"
    elif "wl" in tracer_combination[0] and "wl" in tracer_combination[1]:
        dtype = "cl_ee"
    return dtype


def get_ells_bandwindow(sfile, tracer_name_1, tracer_name_2, ellmax):
    dtype = get_data_type((tracer_name_1, tracer_name_2))
    idx = sfile.indices(data_type=dtype, tracers=(tracer_name_1, tracer_name_2))
    window = sfile.get_bandpower_windows(idx)
    fine_ells = window.values
    indices = (fine_ells >= 2) & (fine_ells <= ellmax)
    fine_ells = jnp.asarray(fine_ells[indices], dtype=jnp.float32)
    bandwindow = jnp.asarray(window.weight[indices])
    return fine_ells, bandwindow


def extract_bandwindow(sfile):
    tracers_names = list(sfile.tracers.keys())
    nbin_gc = sum(["DESgc__" in tracers_names[i] for i in range(len(tracers_names))])
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])

    # galaxy-galaxy
    record_gc = []
    for i in range(nbin_gc):
        tracer_name = f"DESgc__{i}"
        ells_gc, bandwindow = get_ells_bandwindow(
            sfile, tracer_name, tracer_name, ELLMAX_GC
        )
        record_gc.append(bandwindow)

    # galaxy-shear
    record_gc_wl = []
    for i in range(nbin_gc):
        for j in range(nbin_wl):
            tracer_name_1 = f"DESgc__{i}"
            tracer_name_2 = f"DESwl__{j}"
            ells_gc_wl, bandwindow = get_ells_bandwindow(
                sfile, tracer_name_1, tracer_name_2, ELLMAX_GC
            )
            record_gc_wl.append(bandwindow)

    # shear-shear
    record_wl = []
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tracer_name_1 = f"DESwl__{i}"
            tracer_name_2 = f"DESwl__{j}"
            ells_wl, bandwindow = get_ells_bandwindow(
                sfile, tracer_name_1, tracer_name_2, ELLMAX_WL
            )
            record_wl.append(bandwindow)

    return (ells_gc, record_gc), (ells_gc_wl, record_gc_wl), (ells_wl, record_wl)


def extract_data_covariance(saccfile):
    tracers_names = list(saccfile.tracers.keys())
    nbin_gc = sum(["DESgc__" in tracers_names[i] for i in range(len(tracers_names))])
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])

    indices = []
    # galaxy-galaxy
    for i in range(nbin_gc):
        tracer_name = f"DESgc__{i}"
        _, _, ind = saccfile.get_ell_cl(
            "cl_00", tracer_name, tracer_name, return_cov=False, return_ind=True
        )
        indices += list(ind)

    # galaxy-shear
    for i in range(nbin_gc):
        for j in range(nbin_wl):
            tracer_name_1 = f"DESgc__{i}"
            tracer_name_2 = f"DESwl__{j}"
            _, _, ind = saccfile.get_ell_cl(
                "cl_0e", tracer_name_1, tracer_name_2, return_cov=False, return_ind=True
            )
            indices += list(ind)

    # shear-shear
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tracer_name_1 = f"DESwl__{i}"
            tracer_name_2 = f"DESwl__{j}"
            _, _, ind = saccfile.get_ell_cl(
                "cl_ee", tracer_name_1, tracer_name_2, return_cov=False, return_ind=True
            )
            indices += list(ind)

    indices = np.array(indices)
    covariance = saccfile.covariance.covmat[indices][:, indices]
    data = saccfile.mean[indices]
    return jnp.array(data), jnp.array(covariance)


def get_index_pairs(nbin1, nbin2=None, auto=False):
    cl_index = list()
    if nbin2 is not None:
        for i in range(nbin1):
            for j in range(nbin2):
                cl_index.append([i, j + nbin1])
    elif auto:
        for i in range(nbin1):
            cl_index.append([i, i])
    else:
        for i in range(nbin1):
            for j in range(i, nbin1):
                cl_index.append([i, j])
    return cl_index


def get_params_vec(cosmo, multiplicative, deltaz, ia_params, bias, deltaz_gc):
    mparam_1, mparam_2, mparam_3, mparam_4 = multiplicative
    dz1, dz2, dz3, dz4 = deltaz
    a_ia_param, eta_param = ia_params
    b1, b2, b3, b4, b5 = bias
    dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5 = deltaz_gc
    return jnp.array(
        [
            cosmo.sigma8,
            cosmo.Omega_c,
            cosmo.Omega_b,
            cosmo.h,
            cosmo.n_s,
            mparam_1,
            mparam_2,
            mparam_3,
            mparam_4,
            dz1,
            dz2,
            dz3,
            dz4,
            a_ia_param,
            eta_param,
            b1,
            b2,
            b3,
            b4,
            b5,
            dz_gc_1,
            dz_gc_2,
            dz_gc_3,
            dz_gc_4,
            dz_gc_5,
        ]
    )


def unpack_params_vec(params):
    cosmo = jc.Cosmology(
        sigma8=params[0],
        Omega_c=params[1],
        Omega_b=params[2],
        h=params[3],
        n_s=params[4],
        w0=-1.0,
        Omega_k=0.0,
        wa=0.0,
    )
    mparam_1, mparam_2, mparam_3, mparam_4 = params[5:9]
    dz1, dz2, dz3, dz4 = params[9:13]
    a_ia_param, eta_param = params[13], params[14]
    b1, b2, b3, b4, b5 = params[15], params[16], params[17], params[18], params[19]
    dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5 = (
        params[20],
        params[21],
        params[22],
        params[23],
        params[24],
    )
    return (
        cosmo,
        [mparam_1, mparam_2, mparam_3, mparam_4],
        [dz1, dz2, dz3, dz4],
        [a_ia_param, eta_param],
        [b1, b2, b3, b4, b5],
        [dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5],
    )


def interpolator(ellnew, ellcoarse, powerspectrum):
    ellnew_log = jnp.log(ellnew)
    ellcoarse_log = jnp.log(ellcoarse)
    powerspectrum_log = jnp.log(powerspectrum)
    ps_interp = jnp.interp(ellnew_log, ellcoarse_log, powerspectrum_log)
    return jnp.exp(ps_interp)


def get_bandpowers_gc(
    bandwindow_ells, bandwindow_matrix, ells_coarse, powerspectra, nbin_gc
):
    recordbandpowers = []
    counter = 0
    for i in range(nbin_gc):
        cls_wl_interp = interpolator(
            bandwindow_ells, ells_coarse, powerspectra[counter]
        )
        bandpowers = bandwindow_matrix[counter].T @ cls_wl_interp
        recordbandpowers.append(bandpowers)
        counter += 1
    return recordbandpowers


def get_bandpowers_gc_wl(
    bandwindow_ells, bandwindow_matrix, ells_coarse, powerspectra, nbin_gc, nbin_wl
):
    recordbandpowers = []
    counter = 0
    for i in range(nbin_gc):
        for j in range(nbin_wl):
            cls_wl_interp = interpolator(
                bandwindow_ells, ells_coarse, powerspectra[counter]
            )
            bandpowers = bandwindow_matrix[counter].T @ cls_wl_interp
            recordbandpowers.append(bandpowers)
            counter += 1
    return recordbandpowers


def get_bandpowers_wl(
    bandwindow_ells, bandwindow_matrix, ells_coarse, powerspectra, nbin_wl
):
    recordbandpowers = []
    counter = 0
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            cls_wl_interp = interpolator(
                bandwindow_ells, ells_coarse, powerspectra[counter]
            )
            bandpowers = bandwindow_matrix[counter].T @ cls_wl_interp
            recordbandpowers.append(bandpowers)
            counter += 1
    return recordbandpowers


def get_gc_powerspectra(parameters, jax_nz_gc):
    (
        cosmo,
        multiplicative,
        deltaz_wl,
        (a_ia_param, eta_param),
        bias,
        deltaz_gc,
    ) = unpack_params_vec(parameters)
    nbin_gc = len(deltaz_gc)
    gc_biases = [jc.bias.constant_linear_bias(bi) for bi in bias]
    nz_gc_sys = [
        jc.redshift.systematic_shift(nzi, dzi, zmax=ZMAX)
        for nzi, dzi in zip(jax_nz_gc, deltaz_gc)
    ]
    probes_gc = [jc.probes.NumberCounts(nz_gc_sys, gc_biases)]
    ells_coarse = jnp.geomspace(ELLMIN, ELLMAX_GC, NELL, dtype=jnp.float32)
    idx_pairs_gc = get_index_pairs(nbin_gc, auto=True)
    ps_gc = jc.angular_cl.angular_cl(
        cosmo, ells_coarse, probes_gc, index_pairs=idx_pairs_gc
    )
    return ps_gc, ells_coarse, nbin_gc


def get_gc_wl_powerspectra(parameters, jax_nz_gc, jax_nz_wl):
    (
        cosmo,
        multiplicative,
        deltaz_wl,
        (a_ia_param, eta_param),
        bias,
        deltaz_gc,
    ) = unpack_params_vec(parameters)
    nbin_gc = len(deltaz_gc)
    nbin_wl = len(deltaz_wl)

    # apply all the systematics here (shifts, multiplicative bias, intrinsic alignment)
    nz_wl_sys = [
        jc.redshift.systematic_shift(nzi, dzi) for nzi, dzi in zip(jax_nz_wl, deltaz_wl)
    ]
    nz_gc_sys = [
        jc.redshift.systematic_shift(nzi, dzi) for nzi, dzi in zip(jax_nz_gc, deltaz_gc)
    ]
    gc_biases = [jc.bias.constant_linear_bias(bi) for bi in bias]
    b_ia = jc.bias.des_y1_ia_bias(a_ia_param, eta_param, 0.62)

    probes_gc = [jc.probes.NumberCounts(nz_gc_sys, gc_biases)]
    probes_wl = [
        jc.probes.WeakLensing(
            nz_wl_sys, ia_bias=b_ia, multiplicative_bias=multiplicative
        )
    ]
    probes_gc_wl = probes_gc + probes_wl

    ells_coarse = jnp.geomspace(ELLMIN, ELLMAX_GC, NELL, dtype=jnp.float32)
    idx_pairs_gc_wl = get_index_pairs(nbin_gc, nbin_wl, auto=False)
    ps_gc_wl = jc.angular_cl.angular_cl(
        cosmo, ells_coarse, probes_gc_wl, index_pairs=idx_pairs_gc_wl
    )
    return ps_gc_wl, ells_coarse, nbin_gc, nbin_wl


def get_wl_powerspectra(parameters, jax_nz_wl):
    cosmo, multiplicative, deltaz_wl, (a_ia_param, eta_param) = unpack_params_vec(
        parameters
    )[0:4]
    nbin_wl = len(deltaz_wl)

    nz_wl_sys = [
        jc.redshift.systematic_shift(nzi, dzi, zmax=ZMAX)
        for nzi, dzi in zip(jax_nz_wl, deltaz_wl)
    ]
    b_ia = jc.bias.des_y1_ia_bias(a_ia_param, eta_param, 0.62)
    probes_wl = [
        jc.probes.WeakLensing(
            nz_wl_sys, ia_bias=b_ia, multiplicative_bias=multiplicative
        )
    ]

    ells_coarse = jnp.geomspace(ELLMIN, ELLMAX_WL, NELL, dtype=jnp.float32)
    idx_pairs_wl = get_index_pairs(nbin_wl, auto=False)
    ps_wl = jc.angular_cl.angular_cl(
        cosmo, ells_coarse, probes_wl, index_pairs=idx_pairs_wl
    )
    return ps_wl, ells_coarse, nbin_wl


def gc_bandpower_calculation(parameters, jax_nz_gc, bandwindow_ells, bandwindow_matrix):
    ps_gc, ells_coarse, nbin_gc = get_gc_powerspectra(parameters, jax_nz_gc)
    gc_bandpowers = get_bandpowers_gc(
        bandwindow_ells, bandwindow_matrix, ells_coarse, ps_gc, nbin_gc
    )
    return gc_bandpowers


def gc_wl_bandpower_calculation(
    parameters, jax_nz_gc, jax_nz_wl, bandwindow_ells, bandwindow_matrix
):
    ps_gc_wl, ells_coarse, nbin_gc, nbin_wl = get_gc_wl_powerspectra(
        parameters, jax_nz_gc, jax_nz_wl
    )
    gc_wl_bandpowers = get_bandpowers_gc_wl(
        bandwindow_ells, bandwindow_matrix, ells_coarse, ps_gc_wl, nbin_gc, nbin_wl
    )
    return gc_wl_bandpowers


def wl_bandpower_calculation(parameters, jax_nz_wl, bandwindow_ells, bandwindow_matrix):
    ps_wl, ells_coarse, nbin_wl = get_wl_powerspectra(parameters, jax_nz_wl)
    wl_bandpowers = get_bandpowers_wl(
        bandwindow_ells, bandwindow_matrix, ells_coarse, ps_wl, nbin_wl
    )
    return wl_bandpowers


def get_bandpowers_probes(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    theory_gc = gc_bandpower_calculation(parameters, jax_nz_gc, bw_gc[0], bw_gc[1])
    theory_gc_wl = gc_wl_bandpower_calculation(
        parameters, jax_nz_gc, jax_nz_wl, bw_gc_wl[0], bw_gc_wl[1]
    )
    theory_wl = wl_bandpower_calculation(parameters, jax_nz_wl, bw_wl[0], bw_wl[1])
    return theory_gc, theory_gc_wl, theory_wl


def get_bandpowers_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    theory_gc, theory_gc_wl, theory_wl = get_bandpowers_probes(
        parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )
    concat_theory_gc = jnp.concatenate(theory_gc)
    concat_theory_gc_wl = jnp.concatenate(theory_gc_wl)
    concat_theory_wl = jnp.concatenate(theory_wl)
    return jnp.concatenate([concat_theory_gc, concat_theory_gc_wl, concat_theory_wl])
