# Forked from KiMotor by Stefano Cottafavi.
# Copyright 2022 Stefano Cottafavi <stefano.cottafavi@gmail.com>.
# Copyright 2026 I-T-C-R-W
# SPDX-License-Identifier: GPL-2.0-only

import math
import numpy as np
from . import kmotor_pro_linalg as kla
import wx

def parallel(r1,r2, dr,th,turns,dir):
        """ Compute layout points for coil with sides always aligned to radius

        Args:
            r1 (int): coil inner radius
            r2 (int): coil outer radius
            dr (int): spacing between coil loops (and also adj. coils)
            th (float): coil trapezoid angle 
            turns (int): number of coil loops (windings)
            dir (int): coil direction, from larger to smaller loop (0:CW normal, 1:CCW rverse)

        Returns:
            matrix, matrix, matrix: corners (excl. arc mids), outer arc mids, inner arc mids
        """

        pts = []
        mds = []
        mdsi = []

        # points 
        c = [0,0,0]
        l0 = np.array([ c, [r1*math.cos(th/2), r1*math.sin(th/2), 0] ])
        l0 = kla.line_offset(l0, -dr)
        
        for turn in range(turns):
            # offset line
            lr = kla.line_offset(l0, -turn*dr)
            # solve corners and order them
            pc1 = kla.circle_line_intersect(lr, c, r1+turn*dr)
            pc1 = pc1[0:2]
            pc2 = kla.circle_line_intersect(lr, c, r2-turn*dr)
            pc2 = pc2[0:2]
            pc3 = np.array([ pc2[0],-pc2[1] ])
            pc4 = np.array([ pc1[0],-pc1[1] ])
            if dir == 0:
                pts.extend([pc1, pc2, pc3, pc4])
            else:
                pts.extend([pc4, pc3, pc2, pc1])
            
            # solve outer and inner mid-points
            pm = np.array([ r2-turn*dr, 0 ])
            pmi = np.array([ r1+turn*dr, 0 ])
            mds.extend([pm])
            mdsi.extend([pmi])

        pm = np.matrix(pts)     # points, excl. arc mids
        mm = np.matrix(mds)     # arc (outer) mids only
        mmi = np.matrix(mdsi)    # arc (inner) mids only

        return pm, mm, mmi

def radial(ri,ro, dr,th,turns,dir):
        """ Compute layout points for coil with sides aligned to the local radial direction

        Every side is a radial line. The step from one turn to the next is made on the
        inner arc (the descending side stays on its own radial line), so neighbouring
        tracks keep at least ``dr`` centre distance everywhere, also for fine pitches.
        Arc mid points are taken at the true angular middle of each arc.

        Args:
            ri (int): coil inner radius
            ro (int): coil outer radius
            dr (int): spacing between coil loops (and also adj. coils)
            th (float): coil trapezoid angle
            turns (int): number of coil loops (windings)
            dir (int): coil direction, from larger to smaller loop (0:CW normal, 1:CCW rverse)

        Returns:
            matrix: corners and arc mid points in the order expected by coil_tracker()
        """
        c = [0, 0, 0]

        def rot(p, a):
            return np.array([p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a)])

        def polar(r, a):
            return np.array([r * math.cos(a), r * math.sin(a)])

        pts = []
        l0 = np.array([c, [ri * math.cos(th / 2), ri * math.sin(th / 2), 0]])
        a_prev = None
        for turn in range(turns):
            lr = kla.line_offset(l0, -dr)
            pc1 = kla.circle_line_intersect(lr, c, ri + turn * dr)[0:2]
            a1 = math.atan2(pc1[1], pc1[0])          # radial line of this turn
            r_in, r_out = ri + turn * dr, ro - turn * dr
            pc2 = polar(r_out, a1)
            pc3 = polar(r_out, -a1)
            # descending side of the previous turn ends on its own radial line
            pc4 = polar(r_in, -(a_prev if a_prev is not None else a1))
            pmo = np.array([r_out, 0.0])
            pmi = np.array([r_in, 0.0])
            if turn == 0:
                pts.extend([pmi, rot(pmi, a1 / 2), pc1, pc2, pmo, pc3])
            elif turn == turns - 1:
                pts.extend([pc4, pmi, pc1, pc2, rot(pmo, a1 / 2), pmo])
            else:
                pts.extend([pc4, pmi, pc1, pc2, pmo, pc3])
            a_prev = a1
            l0 = [[pc1[0], pc1[1], 0], [pc2[0], pc2[1], 0]]

        if dir != 0:   # reverse direction = mirror image
            pts = [np.array([q[0], -q[1]]) for q in pts]
        return np.matrix(pts)

def compact(ri, ro, dr, th, turns, dir):
        """Dense variant; shares the radial() layout (same point contract)."""
        return radial(ri, ro, dr, th, turns, dir)
