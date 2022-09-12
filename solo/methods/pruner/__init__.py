from .sparse import Mask, CosineDecay, MaskStat
from .contrastive_sparse import ContrastiveMask
from .contrastive_regrow import ContrastiveRegMask
from .contrastive_lamp import ContrastiveLAMP
from .contrastive_momentum_sparse import ContrastiveMMask
from .contrastive_nm_sparse import ContrastiveNM
from .contrastive_momentum_nm_sparse import ContrastiveMNM

from .duo_sparse import DuoMask
from .duo_momentum_sparse import DuoMMask
from .duo_nm_sparse import DuoNMMask
from .duo_nm_momentum_sparse import DuoNMMMask

from .utils import *