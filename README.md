```python
import numpy as np
from rays import *

ANGLE = np.pi / 8

system = System()
system.add_surface(FreeSpace(0, 10, None))
system.add_surface(ThinLens(10, 100, 25))
system.add_surface(FreeSpace(0, 100, None))
[system.launch_ray(y0 - np.tan(ANGLE) * 10, ANGLE, 532) for y0 in np.linspace(-10, 10, 128)]
system.trace()
display_system(system)    
````
![image](https://github.com/sstucker/rays/blob/main/rays.png)
