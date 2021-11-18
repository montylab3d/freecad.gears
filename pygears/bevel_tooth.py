# -*- coding: utf-8 -*-
# ***************************************************************************
# *                                                                         *
# * This program is free software: you can redistribute it and/or modify    *
# * it under the terms of the GNU General Public License as published by    *
# * the Free Software Foundation, either version 3 of the License, or       *
# * (at your option) any later version.                                     *
# *                                                                         *
# * This program is distributed in the hope that it will be useful,         *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of          *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
# * GNU General Public License for more details.                            *
# *                                                                         *
# * You should have received a copy of the GNU General Public License       *
# * along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
# *                                                                         *
# ***************************************************************************

from __future__ import division
from __future__ import division
from numpy import cos, sin, tan, arcsin, arccos, arctan, pi, array, ndarray, linspace, transpose, vstack, sqrt
from ._functions import rotation, reflection, trimfunc, nearestpts, intersection_line_circle


class BevelTooth(object):
    def __init__(self, pressure_angle=20 * pi / 180, pitch_angle=pi / 4, clearance=0.25,
                 shift=0.0, z=21, backlash=0.00, module=1.0):
        self.pressure_angle = pressure_angle
        self.pitch_angle = pitch_angle
        self.z = z
        self.clearance = clearance
        self.backlash = backlash
        self.module = module
        self.shift = shift
        self.addendum = 1.
        self.dedendum = 1.

        # pitch diameter
        self.d = self.z * self.module

        # pitch radius
        self.r = self.d * 0.5

        # cone distance, aka, radius of development sphere
        self.R0 = self.r / sin(self.pitch_angle)

        # addendum angle
        self.theta_a = arctan((self.addendum + shift) * module / self.R0)
        
        # dedendum angle
        self.theta_f = arctan((self.dedendum + clearance - shift) * module / self.R0)

        # base circle angle
        self.phi_b = arcsin(sin(self.pitch_angle) * cos(self.pressure_angle))
        
        # outer cone angle
        self.phi_a = self.pitch_angle + self.theta_a

        # root cone angle
        self.phi_f = self.pitch_angle - self.theta_f

        # distance from cone apex to center of gear at pitch diameter
        self.X = self.R0 * cos(self.pitch_angle)

        # roll angle of involute at addendum
        self.involute_end = self.inverse_involute(self.phi_a)
        
        # involute generation starting angle; starts at the shallower of base or foot
        if self.phi_b <= self.phi_f:
            self.involute_start = self.inverse_involute(self.phi_f)
        else:
            self.involute_start = 0.

        gear_tooth_p = pi / z + shift * module * tan(pressure_angle)

        # rotation used to center the tooth, aka the absolute angle
        # roll angle of involute at pitch cone
        pitch_roll = self.inverse_involute(self.pitch_angle)
        # involute inclination at pitch circle
        pitch_phi = pitch_roll*sin(self.phi_b)
        # involute azimuth at pitch circle
        pitch_azimuth = pitch_roll - arctan(tan(pitch_phi)/sin(self.phi_b))
        self.involute_rot = gear_tooth_p/2 + pitch_azimuth

        # roll angle limits for generation of the trochoidal portion of
        # the gear root
        self.trochoid_start = 0. # start at the beginning
        # trochoid ends at the form profile, but the other gear
        # implementations use the addendum as the computation limit.
        # I have no problem with that.
        self.trochoid_end = self.inverse_trochoid(self.phi_a)
        # rotation angle to center trochoidal portion of tooth root
        # width of the tooth root?  well, we gotta find it...
        b = arctan((self.dedendum + clearance) * module / self.R0)
        B = pi/2 - self.pressure_angle        
        C = pi/2
        # we have two angles and one side.  First, another side by law of sines
        c = arcsin(sin(b)/sin(B))
        # now side a by right angle law of cosines
        Rd = sin(self.pitch_angle)
        a = arccos(cos(c)/cos(b))
        P = pi/self.z/2 - a/Rd
        
        self.trochoid_rot = pi/self.z - P

                
        print("m="+str(self.module))
        print("z="+str(self.z))
        print("theta_a="+str(self.theta_a/pi*180))
        print("phi_a="+str(self.phi_a/pi*180))

        print("phi_b="+str(self.phi_b/pi*180))
        
        print("theta_f="+str(self.theta_f/pi*180))
        print("phi_f="+str(self.phi_f/pi*180))
        
        print("involute_start="+str(self.involute_start/pi*180))
        print("involute_end="+str(self.involute_end/pi*180))
        print("involute_rot="+str(self.involute_rot/pi*180))
        print("trochoid_end="+str(self.trochoid_end/pi*180))
        print("trochoid_rot="+str(self.trochoid_rot/pi*180))
        print("a="+str(a))
        print("b="+str(b))
        print("c="+str(c))
        print("B="+str(B))
        print("C="+str(C))
        print("P="+str(C))
        print("R0="+str(self.R0))
        print("X="+str(self.X))

    def xyz(azimuth,inclination,R):
        Rxy = R*sin(inclination)
        x = Rxy*cos(azimuth)
        y = Rxy*sin(azimuth)
        z = R*cos(inclination)
        return([x,y,z])
    
    def plot(Az, Ai, Bz, Bi):
        A = self.xyz(Az,Ai,1.)
        B = self.xyz(Bz,Bi,1.)
        print(str(A[0])+" "+str(A[1])+"\n"+str(B[0])+" "+str(B[1])+"\n\n")
        
    def involute_function(self):
        def func(roll_angle):
            # straight-line radius of the gear's base circle on a unit generating sphere
            Rb = sin(self.phi_b)
            # arc distance the rack has 'rolled' around edge of base circle.
            AB = roll_angle*Rb
            # the same distance has been rolled along the great circle of the rack
            BC = AB
            # great circle chord length from gear axis to gear base circle
            OB = self.phi_b
            # great circle chord length from gear axis to involute point, by spherical
            # right triangle law of cosines
            OC = arccos(cos(BC)*cos(OB))
            # angle between OB and OC at gear axis, by spherical right
            # triangle sine identity
            alpha = arcsin(sin(BC)/sin(OC))
            # location of involute point on surface of generating unit sphere in
            # spherical coordinates
            inclination = OC
            azimuth = roll_angle - alpha
            # scale unit sphere by R0, and decompose into cartesian x/y/z
            Rxy = self.R0*sin(inclination)
            x = Rxy*cos(azimuth)
            y = Rxy*sin(azimuth)
            z = self.R0*cos(inclination)
            print(str(x/z)+" "+str(y/z))
            # conical projection to z=1
            return([x/z,y/z])
        return(func)

    def inverse_involute(self, inclination):
        roll_angle = arccos(cos(inclination)/cos(self.phi_b)) / sin(self.phi_b)
        return (roll_angle)

    def involute_points(self, num=10):
        pts = linspace(self.involute_start, self.involute_end, num=num)
        fn = self.involute_function()
        xy = array(list(map(fn, pts)))
        rot = rotation(self.involute_rot - self.backlash / 4)        
        xy = rot(xy)
        print(str(xy))
        return(xy)

    def trochoid_function(self):
        def func(roll_angle):
            # This tracks a physical point (vertex of the rack
            # involute face and the rack top land) to trace the
            # trochoid pattern of the generated gear root.  Unlike
            # above, the rack and gear both roll on theit pitch
            # diameters, not the gear's base diameter.

            # This calculation can be handled almost identically to
            # the involute generation (just hang an extra vertex off
            # the side, CDO) but then it's more complicated than it
            # needs to be. Instead, we extend CD and BO to meet at point
            # P, forming a semilunar triangle BPC with two right
            # angles.
            
            # Straight line radius of gear's pitch circle
            Rpitch = sin(self.pitch_angle)
            # great circle chord length from gear axis to gear pitch
            # circle
            BO = self.pitch_angle
            # CD is equal to the dedendum delta angle
            CD = self.theta_f
            # distance the rack has 'rolled' around edge of pitch
            # circle.
            AB = roll_angle*Rpitch
            # the same distance has been rolled along the pitch circle
            # of the rack
            BC = AB
            # BC, the base of the semilune, is also the same as angle
            # BPC.
            BPC = BC
            # CP=pi/2, BP=PI/2.
            CP = pi/2
            BP = pi/2
            # we know BO and CD, OP and DP are the remaining distances
            # on the same leg
            OP = BP - BO
            DP = CP - CD
            # solve for DO using spherical law of cosines
            DO = arccos(cos(OP)*cos(DP) + sin(OP)*sin(DP)*cos(BPC))
            inclination = DO
            # azimuth still needed
            if (roll_angle == 0):
                # round off error is dangerous to the arccos below
                # when at roll_angle==0, but we know the azimuth is 0
                # in this case anyway
                azimuth = 0
            else:
                BD = arccos(cos(CD)*cos(BC))
                # rearranged law of cosines (solving for the angle)
                BOD = arccos((cos(BD) - cos(BO)*cos(DO))/(sin(BO)*sin(DO)))
                azimuth = BOD-roll_angle
            # scale unit sphere by R0, and decompose into cartesian x/y/z
            Rxy = self.R0*sin(inclination)
            x = Rxy*cos(azimuth)
            y = Rxy*sin(azimuth)
            z = self.R0*cos(inclination)
            # conical projection to z=1
            print(str(x/z)+" "+str(y/z))
            return([x/z,y/z])
        return(func)

    def inverse_trochoid(self, inclination):
        Rpitch = sin(self.pitch_angle)
        BO = self.pitch_angle
        CD = self.theta_f
        CP = pi/2
        BP = pi/2
        OP = BP - BO
        DP = CP - CD
        DO = inclination
        BPC = arccos((cos(DO) - cos(OP)*cos(DP)) / (sin(OP)*sin(DP)))
        BC = BPC
        AB = BC
        roll_angle = AB/Rpitch
        return (roll_angle)

    def trochoid_points(self, num=10):
        pts = linspace(self.trochoid_start, self.trochoid_end, num=num)
        fn = self.trochoid_function()
        xy = array(list(map(fn, pts)))
        rot = rotation(self.trochoid_rot - self.backlash / 4)        
        xy = rot(xy)
        print(str(xy))
        return(xy)
                             
    def points(self, num=10):
        l1 = self.trochoid_points(num=num)
        l2 = self.involute_points(num=num)
        #if self.undercut:
            # if there's undercut, l1 will intersect l2, and trimfunc will
            # return a composite wire consisting of involute and undercut
            # from the trochoid.            
        s = trimfunc(l1, l2[::-1])
        if isinstance(s, ndarray):
            u1, e1 = s
        else:
            # no intersection, no undercut.  Look for the closest
            # approach and transisiton to the trochoid section to
            # fillet the root, like in the involute spur gears.
            u1, e1 = nearestpts(l2, l1)
        #else:
        #    u1 = False
        #    if self.phi_b > self.phi_f:
        #        u1 = vstack(
        #            [[l2[0] * self.phi_f / (diff_norm(l2[0], [0, 0]) * 2)], [l2[0]]])
        #        e1 = l2
        #    else:
        #        e1 = l2

        reflect = reflection(0)
        e2 = reflect(e1)[::-1]
        if isinstance(u1, bool):
            one_tooth = [e1, [e1[-1], e2[0]], e2]
        else:
            u2 = reflect(u1)[::-1]
            one_tooth = [u1, e1, [e1[-1], e2[0]], e2, u2]

        print(str(one_tooth))

        # add a Z dimension (all 1.) for use by the upper layer
        xyz = []
        for wire in one_tooth:
            xyzwire=[]
            for point in wire:
                x = point[0]
                y = point[1]
                xyzwire.append(array([x, y, 1.]))
            xyz.append(array(xyzwire))
        return(xyz)

    def _update(self):
        self.__init__(z=self.z, clearance=self.clearance,
                      pressure_angle=self.pressure_angle,
                      pitch_angle=self.pitch_angle,
                      backlash=self.backlash, module=self.module)
