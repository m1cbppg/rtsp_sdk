"""Reference-based detail-loss guard; uncertainty is not a corruption diagnosis."""
import cv2
import numpy as np


class DetailLossGuard:
    def __init__(self,reference):
        self.size=(640,round(reference.shape[0]*640/reference.shape[1]))
        self.reference_std=self._features(reference)[2]

    def _features(self,frame):
        small=cv2.resize(frame,self.size,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean=cv2.blur(gray,(9,9))
        std=np.sqrt(np.maximum(0,cv2.blur(gray*gray,(9,9))-mean*mean))
        return small,mean,std

    def inspect(self,frame):
        small,mean,std=self._features(frame)
        neutral=(small.max(axis=2).astype(int)-small.min(axis=2))<6
        lost=((std<2)&(self.reference_std>6)&neutral&(mean>50)&(mean<220)).astype(np.uint8)
        # Bridge small residual codec texture, not entire objects or ROIs.
        connected=cv2.morphologyEx(lost,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(connected,8)
        largest=int(max(stats[1:,cv2.CC_STAT_AREA],default=0))
        fraction=float(lost.mean());largest_fraction=largest/lost.size
        return {'detail_loss_fraction':round(fraction,5),
                'largest_detail_loss_fraction':round(largest_fraction,5),
                # Whole-frame abstention only for broad detail loss. Ordinary
                # smaller person/vehicle occlusion remains a local actor check.
                'usable':not (fraction>.25 and largest_fraction>.30)}
