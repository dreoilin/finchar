import numpy as np
import json
import ast
import configparser
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple

def matrange(start, step, stop):
    num = round((stop - start) / step + 1)
    return np.linspace(start, stop, num)

def toupper(optionstr: str) -> str:
    return optionstr.upper()

@dataclass
class SweepConfig(ABC):
    config_file_path: str
    _configParser: configparser.ConfigParser = field(default_factory=configparser.ConfigParser, repr=False)
    _config: dict = field(init=False)

    def __post_init__(self):
        self._configParser.optionxform = toupper    
        self._configParser.read(self.config_file_path)
        self._config = {s:dict(self._configParser.items(s)) for s in self._configParser.sections()}
        self._parse_ranges()
        
        # Generate both the subcircuit parameters file and the main netlist
        self._generate_params_file()
        with open('pysweep.scs', 'w') as netlist_file:
            netlist_file.write(self._generate_netlist())

        self._config['outvars'] =     ['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']
        self._config['outvars_noise'] = ['STH','SFL']
        n, p, n_noise, p_noise = self._generate_outvars()
        self._config['n'] = n
        self._config['p'] = p
        self._config['n_noise'] = n_noise
        self._config['p_noise'] = p_noise

    def __getitem__(self, key):
        if key not in self._config.keys():
            raise ValueError(f"Lookup table does not contain this data")
    
        return self._config[key]
        
    def _parse_ranges(self):
        # Scan for sweep ranges, avoiding the legacy 'LENGTH' key
        for k in ['VGS', 'VDS', 'VSB', 'NS']:
            if k in self._config['SWEEP']:
                v = ast.literal_eval(self._config['SWEEP'][k])
                v = [v] if type(v) is not list else v
                # Check if it's a range tuple to expand, otherwise keep as is
                if isinstance(v[0], (Tuple, list)):
                    v = [matrange(*r) for r in v]
                    v = [val for r in v for val in r]
                self._config['SWEEP'][k] = v
        
        # Ensure geometry parameters are integers
        for k in ['NFINS', 'NFING']:
            if k in self._config['SWEEP']:
                self._config['SWEEP'][k] = int(self._config['SWEEP'][k])
    
    def generate_m_dict(self):
        return {
            'INFO' : self._config['MODEL']['INFO'],
            'CORNER' : self._config['MODEL']['CORNER'],
            'TEMP' : float(self._config['MODEL']['TEMP']),
            'NFING' : self._config['SWEEP']['NFING'],
            'NS' : np.array(self._config['SWEEP']['NS']).T, 
            'NFINS' : self._config['SWEEP'].get('NFINS', 1),
            'VGS' : np.array(self._config['SWEEP']['VGS']).T,
            'VDS' : np.array(self._config['SWEEP']['VDS']).T,
            'VSB' : np.array(self._config['SWEEP']['VSB']).T 
        }
    
    def _write_params(self, **kwargs):
        paramfile = self._config['MODEL'].get('PARAMFILE', 'params.scs')
        with open(paramfile, 'w') as outfile:
            outfile.write(f"parameters {' '.join([f'{k}={v}' for k, v in kwargs.items()])}")
            
    def _build_subckt_string(self, subckt_name: str, model_name: str, stacks: int) -> str:
        """ Helper method to dynamically construct stacked FinFET subcircuits """
        lines = [
            f"subckt {subckt_name} D G S B",
            "parameters l=18n nfin=2 nf=1 asej=6.528e-15 adej=6.528e-15 psej=2.32e-07 \\",
            "        pdej=2.32e-07 lrsd=18n"
        ]
        
        # Iteratively wire the stack from top (drain) to bottom (source)
        for i in range(stacks - 1, -1, -1):
            drain_node = 'D' if i == stacks - 1 else f'n{i+1}'
            source_node = 'S' if i == 0 else f'n{i}'
            
            # Instantiating the device; overrides for nfin/nf are handled in the main netlist
            inst_str = f"    S{i} ({drain_node} G {source_node} B) {model_name} l=l nfin=nfin nf=nf m=1 ngcon=1 \\"
            param_str = "        asej=asej adej=adej psej=psej pdej=pdej lrsd=lrsd"
            lines.extend([inst_str, param_str])
            
        lines.append(f"ends {subckt_name}")
        return '\n'.join(lines)

    def _generate_params_file(self, ns=None):
        """ 
        Generates the params.scs file. 
        If ns_override is provided, use it; otherwise, default to the first value.
        """
        paramfile = self._config['MODEL'].get('PARAMFILE', 'params.scs')
        
        if ns is not None:
            stacks = int(ns)
        else:
            # Fallback: safely extract the first value from the NS list
            ns_data = self._config['SWEEP']['NS']
            stacks = int(ns_data[0] if isinstance(ns_data, list) else ns_data)
        
        nmos_subckt = self._build_subckt_string("mn_s", self._config['MODEL']['MODELN'], stacks)
        pmos_subckt = self._build_subckt_string("mp_s", self._config['MODEL']['MODELP'], stacks)
        
        # ... (rest of the content generation remains the same)
        content = '\n'.join((
            "// params.scs - Auto-generated by Python",
            "simulator lang=spectre",
            "\n// NMOS Subcircuit",
            nmos_subckt,
            "\n// PMOS Subcircuit",
            pmos_subckt,
            ""
        ))
        
        with open(paramfile, 'w') as f:
            f.write(content)

    @abstractmethod
    def _generate_netlist(self) -> str:
        """ Generate the main netlist for the simulation. """
        modelfile = self._config['MODEL']['FILE']
        paramfile = self._config['MODEL'].get('PARAMFILE', 'params.scs')
        temp = float(self._config['MODEL']['TEMP'])-273.15
        
        VDS_max = max(self._config['SWEEP']['VDS'])
        VDS_step = self._config['SWEEP']['VDS'][1] - self._config['SWEEP']['VDS'][0] 
        VGS_max = max(self._config['SWEEP']['VGS'])
        VGS_step = self._config['SWEEP']['VGS'][1] - self._config['SWEEP']['VGS'][0]
        
        # Extract geometry constraints
        NFINS = self._config['SWEEP']['NFINS']
        NFING = self._config['SWEEP']['NFING']
    
        return '\n'.join((
            f"// pysweep.scs - Auto-generated main execution netlist",
            f"simulator lang=spectre",
            f"global 0",
            f"",
            f"parameters ds=0.8 gs=0.0",
            f"",
            f"include {modelfile} section=tt",
            f"include \"{paramfile}\"",
            f"include \"vsb.scs\"",
            f"",
            f"// Device Instantiations (L=18.0n fixed, geometry parametrised by Python)",
            f"mn (vdn vgn vsn vbn) mn_s m=1 l=18.0n nfin={NFINS} nf={NFING} \\",
            f"        asej=6.528e-15 adej=6.528e-15 psej=2.32e-07 pdej=2.32e-07 lrsd=68.0n",
            f"mp (vdp vgp vsp vbp) mp_s m=1 l=18.0n nfin={NFINS} nf={NFING} \\",
            f"        asej=6.528e-15 adej=6.528e-15 psej=2.32e-07 pdej=2.32e-07 lrsd=68.0n",
            f"",
            f"// PMOS Ports and Dummy Probe",
            f"PORTPG (vgp 0) port r=50 dc=-gs",
            f"PORTPS (vsp 0) port r=50 dc=0",
            f"PORTPB (vbp 0) port r=50 dc=sb",
            f"PORTPD (vdp_port 0) port r=50 dc=-ds",
            f"vx_p (vdp vdp_port) vsource dc=0 type=dc",
            f"",
            f"// NMOS Ports and Dummy Probe",
            f"PORTNB (vbn 0) port r=50 dc=-sb",
            f"PORTND (vdn_port 0) port r=50 dc=ds",
            f"vx_n (vdn vdn_port) vsource dc=0 type=dc",
            f"PORTNS (vsn 0) port r=50 dc=0",
            f"PORTNG (vgn 0) port r=50 dc=gs",
            f"",
            f"simulatorOptions options psfversion=\"1.4.0\" reltol=1e-5 vabstol=1e-9 \\",
            f"    iabstol=1e-15 temp={temp} tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \\",
            f"    maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \\",
            f"    redefinedparams=warning \\",
            f"    sensfile=\"../psf/sens.output\" checklimitdest=sqldb",
            f"",
            f"// -------------------------------------------------------------------",
            f"// Two-Dimensional Sweep Setup (NMOS & PMOS Characterisation)",
            f"// -------------------------------------------------------------------",
            f"sweep_vds sweep param=ds start=0 stop={VDS_max} step={VDS_step} {{",
            f"    sweepvgs dc param=gs start=0 stop={VGS_max} step={VGS_step}",
            f"    ",
            f"    // --- NMOS Analyses ---",
            f"    sp_nmos sp ports=[PORTNG PORTNS PORTND PORTNB] freq=1 \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"    ",
            f"    noise_nmos_fl noise freq=1 oprobe=vx_n \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"    ",
            f"    noise_nmos_th noise freq=10G oprobe=vx_n \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"",
            f"    // --- PMOS Analyses ---",
            f"    sp_pmos sp ports=[PORTPG PORTPS PORTPD PORTPB] freq=1 \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"    ",
            f"    noise_pmos_fl noise freq=1 oprobe=vx_p \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"    ",
            f"    noise_pmos_th noise freq=10G oprobe=vx_p \\",
            f"        param=gs start=0 stop={VGS_max} step={VGS_step} annotate=status",
            f"}}",
            f"",
            f"saveOptions options save=allpub"
        ))
    
    @abstractmethod
    def _generate_outvars(self, n: list=[], p: list=[], n_noise: list=[], p_noise: list=[]) -> Tuple[list, list, list, list]:
        """ Generate the mapping of output variables from the simulation to the lookup table. """
        n.append( ['mn:ids','A',       [1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:vth','V',       [0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:igd','A',       [0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:igs','A',       [0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gm','S',        [0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gmbs','S',      [0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gds','S',       [0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgg','F',       [0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgs','F',       [0,    0,   0,    0,    0,   0,    0,    0,   -1,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgd','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,   -1,    0,    0,    0,    0  ]])
        n.append( ['mn:cgb','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,   -1,    0,    0  ]])
        n.append( ['mn:cdd','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['mn:cdg','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,   -1,    0,    0,    0  ]])
        n.append( ['mn:css','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        n.append( ['mn:csg','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,   -1,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cjd','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['mn:cjs','F',       [0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])

        p.append( ['mp:ids','A',       [-1,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:vth','V',       [ 0,   -1,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:igd','A',       [ 0,    0,   -1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:igs','A',       [ 0,    0,    0,   -1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gm','S',        [ 0,    0,    0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gmbs','S',      [ 0,    0,    0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gds','S',       [ 0,    0,    0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgg','F',       [ 0,    0,    0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgs','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,   -1,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgd','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,   -1,    0,    0,    0,    0  ]])
        p.append( ['mp:cgb','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,   -1,    0,    0  ]])
        p.append( ['mp:cdd','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['mp:cdg','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,   -1,    0,    0,    0  ]])
        p.append( ['mp:css','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        p.append( ['mp:csg','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,   -1,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cjd','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['mp:cjs','F',       [ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        
        # Outnoise correctly maps the raw noise power measured directly at the drain
        n_noise.append(['noise_nmos_fl:out', 'A^2/Hz', [0, 1]])
        n_noise.append(['noise_nmos_th:out', 'A^2/Hz', [1, 0]])
        
        p_noise.append(['noise_pmos_fl:out', 'A^2/Hz', [0, 1]])
        p_noise.append(['noise_pmos_th:out', 'A^2/Hz', [1, 0]])
        
        return (n, p, n_noise, p_noise)

class Config(SweepConfig):
    """ Configuration class for the sweep simulation. """
    def __post_init__(self):
        super().__post_init__()
    
    def _generate_netlist(self):
        return super()._generate_netlist()
    
    def _generate_outvars(self, *args, **kwargs):
        return super()._generate_outvars(*args, **kwargs)
