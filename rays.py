# -*- coding: utf-8 -*-
"""

Simple matrix paraxial raytracer

Created on Thu Oct 26 13:25:06 2023

@author: sstucker
"""

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


class Surface:
    
    def __init__(self, position):
        self.position = position
        self.thickness = 0
    
    @property
    def H(self):
        raise NotImplementedError()
        

class ThinLens(Surface):
    
    def __init__(self, position, focal_length, aperture):
        super().__init__(position)
        self.focal_length = focal_length
        self.aperture = aperture
        
    @property
    def H(self):
        return np.array([[1, 0], [-1/self.focal_length, 1]])


class FreeSpace(Surface):
    
    def __init__(self, position, distance, aperture):
        super().__init__(position)
        self.thickness = distance
        self.aperture = aperture
        
    @property
    def H(self):
        return np.array([[1, self.thickness], [0, 1]])


class Ray:
    
    def __init__(self, y0: float, θ0: float, λ: float):
        self._s = np.array([[y0], [θ0]])
        self._x = 0
        self.wavelength = λ
        self._points = []
        self._points.append((0, y0))
        self._vignetted = False
        
    def is_vignetted(self):
        return self._vignetted
        
    @property
    def s(self):
        return self._s
 
    @property
    def points(self):
        return np.array(self._points, dtype=object)  
 
    @property
    def height(self):
        return self._s[0] 

    @property
    def angle(self):
        return self._s[1]     

    def transfer(self, surface):
        self._s = np.matmul(surface.H, self.s)
        self._x += surface.thickness
        if not self._vignetted:
            self._points.append((self._x, self._s[0]))
        if surface.aperture is not None and np.abs(self._s[0]) > surface.aperture / 2:
            self._vignetted = True
        

def trace_ray(ray, surfaces) -> Ray:
    for surface in surfaces:
        ray.transfer(surface)
    return ray


class System:
    
    def __init__(self):
        self._surfaces = []
        self._rays = []
        self._traced = False
        
    def add_surface(self, surface: Surface):
        self._surfaces.append(surface)
        # isorted = np.argsort([s.position for s in self._surfaces])
        # self._surfaces = [self._surfaces[i] for i in isorted]
        self._traced = False
        
    def launch_ray(self, y0: float, θ0: float, λ: float, id=None):
        self._rays.append(Ray(y0, θ0, λ))
        self._traced = False
        
    def reset_rays(self):
        self._rays = []
        self._traced = False
    
    def trace(self, use_multiprocessing=False):
        if use_multiprocessing is True:
            # This is never faster as far as I can tell
            pool = mp.Pool(processes=os.cpu_count())          
            results = [pool.apply_async(trace_ray, args=(ray, self._surfaces)) for ray in self._rays]
            self._rays = [result.get() for result in results]
        else:
            self._rays = [trace_ray(ray, self._surfaces) for ray in self._rays]
        self._traced = True
    
    @property
    def pupil_diameter(self):
        return self._surfaces[0].aperture
    
    @property
    def rays(self):
        return self._rays 
    
    @property
    def surfaces(self):
        return self._surfaces
    
    def proportion_vignetted(self) -> float:
        return sum([int(ray.is_vignetted()) for ray in self._rays]) / len(self._rays)


def display_system(system, draw_vignetted=True):
    plt.figure('Optical system display')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    for surface in system.surfaces:
        if surface.__class__.__name__ == 'ThinLens':
            for direction in [-1, 1]:
                plt.arrow(surface.position, 0, 0, direction * surface.aperture / 2, head_length=surface.aperture / 16, head_width=surface.aperture / 16, facecolor='black')      
    for ray in system.rays:
        if draw_vignetted:
            plt.plot(ray.points[:, 0], ray.points[:, 1], linewidth=0.3, alpha=0.2, color='red')
        else:
            if not ray.is_vignetted():
                plt.plot(ray.points[:, 0], ray.points[:, 1], linewidth=0.3, alpha=0.2, color='red')


if __name__ == '__main__':
    
    ANGLE = np.pi / 8
    
    system = System()
    system.add_surface(FreeSpace(0, 10, None))
    system.add_surface(ThinLens(10, 100, 25))
    system.add_surface(FreeSpace(0, 100, None))
    [system.launch_ray(y0 - np.tan(ANGLE) * relay.relay_to_image_distance, ANGLE, 532) for y0 in np.linspace(-10, 10, 128)]
    system.trace()
    display_system(system)    
