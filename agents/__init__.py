from agents.crl_infonce import CRLInfoNCEAgent
from agents.dino_rebrac import DINOReBRACAgent
from agents.fb_repr import ForwardBackwardRepresentationAgent
from agents.hilp import HILPAgent
from agents.infom import InFOMAgent
from agents.infom_multimodal import InFOMMultiModalAgent
from agents.infom_state_decoder import InFOMStateDecoderAgent
from agents.infom_lang_state_decoder import InFOMLangStateDecoderAgent
from agents.infom_dino_attnpool import InFOMDinoAttnPoolAgent
from agents.iql import IQLAgent
from agents.mbpo_rebrac import MBPOReBRACAgent
from agents.rebrac import ReBRACAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl_infonce=CRLInfoNCEAgent,
    dino_rebrac=DINOReBRACAgent,
    fb_repr=ForwardBackwardRepresentationAgent,
    hilp=HILPAgent,
    infom=InFOMAgent,
    infom_multimodal=InFOMMultiModalAgent,
    infom_state_decoder=InFOMStateDecoderAgent,
    infom_lang_state_decoder=InFOMLangStateDecoderAgent,
    infom_dino_attnpool=InFOMDinoAttnPoolAgent,
    iql=IQLAgent,
    mbpo_rebrac=MBPOReBRACAgent,
    rebrac=ReBRACAgent,
    td_infonce=TDInfoNCEAgent,
)
