from __future__ import print_function
import numpy as np
from numpy import pi
from . import model

class EulerModel(model.Model):
    """2D vorticity model. This is just 2D Euler."""
    
     def __init__(
        self,
        **kwargs
        ):
            
        # initial conditions: (PV anomalies)
        self.set_q(1e-3*np.random.rand(1,self.ny,self.nx))

     ### PRIVATE METHODS - not meant to be called by user ###
    
    def _initialize_background(self):
        pass
    
    def _initialize_inversion_matrix(self):
        """ the inversion to go from the FT of the vorticity to the FT of the stream function"""
        # The inversion comes from qh = -kappa**2 ph
        self.a = -np.asarray(self.wv2i)[np.newaxis, np.newaxis, :, :]
        # np.newaxis increases the dimension of the array so it fits with the general definition of an 
        # inversion matrix a. The two extra dimensions are for when there are layers. 
        # wv2i is 1 / k^2+l^2 
        
    def _initialize_forcing(self):
        pass
    
    def _initialize_forcing(self):
        pass
    
    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg, self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    def _calc_ke(self):
        ke = .5*self.spec_var(self.wv*self.ph)
        return ke.sum()

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = .5*self.H * spec_var(self, self.wv2*self.ph)
        return 2.*pi*np.sqrt( self.H / ens ) / year