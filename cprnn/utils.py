import numpy as np


def convert_ori_rep(ori_dict, type_out="euler"):
    if type_out != "euler" and ori_dict.get("type") != "quat":
        raise NotImplementedError("Can only convert euler to quat.")

    assert ori_dict.get("quat_component_ordering") == "scalar-vector"

    return {
        "type": "euler",
        "unit_cell_alignment": ori_dict.get("unit_cell_alignment"),
        "euler_degrees": False,
        "eulers": qu2eu(ori_dict["quaternions"], P=ori_dict.get("P", -1)),
    }  # orientation_coordinate_system??


################################################################################
# Code below available according to the following conditions on 
# https://github.com/MarDiehl/3Drotations
################################################################################
# Copyright (c) 2017-2020, Martin Diehl/Max-Planck-Institut für Eisenforschung GmbH
# Copyright (c) 2013-2014, Marc De Graef/Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     - Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     - Neither the names of Marc De Graef, Carnegie Mellon University nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
def qu2eu(qu: np.ndarray, P=1) -> np.ndarray:
    """
    Quaternion to Bunge Euler angles.

    References
    ----------
    E. Bernardes and S. Viollet, PLoS ONE 17(11):e0276302, 2022
    https://doi.org/10.1371/journal.pone.0276302

    Source
    ------
    https://github.com/eisenforschung/DAMASK/blob/release/python/damask/_rotation.py

    """
    a = qu[..., 0:1]
    b = -P * qu[..., 3:4]
    c = -P * qu[..., 1:2]
    d = -P * qu[..., 2:3]

    eu = np.block(
        [
            np.arctan2(b, a),
            np.arccos(2 * (a**2 + b**2) / (a**2 + b**2 + c**2 + d**2) - 1),
            np.arctan2(-d, c),
        ]
    )

    eu_sum = eu[..., 0] + eu[..., 2]
    eu_diff = eu[..., 0] - eu[..., 2]

    is_zero = np.isclose(eu[..., 1], 0.0)
    is_pi = np.isclose(eu[..., 1], np.pi)
    is_ok = ~np.logical_or(is_zero, is_pi)

    eu[..., 0][is_zero] = 2 * eu[..., 0][is_zero]
    eu[..., 0][is_pi] = -2 * eu[..., 2][is_pi]
    eu[..., 2][~is_ok] = 0.0
    eu[..., 0][is_ok] = eu_diff[is_ok]
    eu[..., 2][is_ok] = eu_sum[is_ok]

    eu[np.logical_or(np.abs(eu) < 1.0e-6, np.abs(eu - 2 * np.pi) < 1.0e-6)] = 0.0
    return np.where(eu < 0.0, eu % (np.pi * np.array([2.0, 1.0, 2.0])), eu)
