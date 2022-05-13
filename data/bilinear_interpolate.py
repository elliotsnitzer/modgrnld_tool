#!/usr/bin/env python

import numpy as np

def bilinear_interpolate(im, x, y, **kwargs):
   if 'nodataValues' in kwargs:
      nodataValues = kwargs['nodataValues']
   else:
      nodataValues = np.nan

   x = np.asarray(x)
   y = np.asarray(y)

   x0 = np.floor(x).astype(int)
   x1 = x0 + 1
   y0 = np.floor(y).astype(int)
   y1 = y0 + 1

   x0 = np.clip(x0, 0, im.shape[1]-1);
   x1 = np.clip(x1, 0, im.shape[1]-1);
   y0 = np.clip(y0, 0, im.shape[0]-1);
   y1 = np.clip(y1, 0, im.shape[0]-1);

   Ia = im[ y0, x0 ]
   Ib = im[ y1, x0 ]
   Ic = im[ y0, x1 ]
   Id = im[ y1, x1 ]

   wa = (x1-x) * (y1-y)
   wb = (x1-x) * (y-y0)
   wc = (x-x0) * (y1-y)
   wd = (x-x0) * (y-y0)

   if np.any(np.abs(Ia - nodataValues)<1e-10) or np.any(np.abs(Ib - nodataValues)<1e-10) or np.any(np.abs(Ic - nodataValues)<1e-10) or np.any(np.abs(Id - nodataValues)<1e-10):
      return np.nan
   else:
      return wa*Ia + wb*Ib + wc*Ic + wd*Id

