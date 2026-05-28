import concurrent.futures
import glob
import multiprocessing as mp
import os
import pickle
import re
import shutil

import numpy as np
import psf_utils
from tqdm import tqdm

from .config import Config
from .simulator import SpectreSimulator

def s2y(S, Z0=50):
    """ Converts an N-port S-parameter matrix (shape: N, N, num_points) to a Y-parameter matrix. """
    # Move the port dimensions to the end for np.linalg to batch process the matrices
    S_swapped = np.moveaxis(S, [0, 1], [-2, -1])
    I = np.eye(S.shape[0])
    
    Y_swapped = (1/Z0) * np.matmul((I - S_swapped), np.linalg.inv(I + S_swapped))
    
    # Return axes to the original shape
    return np.moveaxis(Y_swapped, [-2, -1], [0, 1])

class Sweep:
    def __init__(self, config_file_path: str):
        self._config = Config(config_file_path)
        spectre_args = ['+escchars', 
                '=log', 
                'spectre.out', 
                '-format', 
                'psfascii', 
                '-raw', 
                'psf']

        self._simulator = SpectreSimulator(*spectre_args)
    
    def run(self):
        
        NSs = self._config['SWEEP']['NS']
        VSBs = self._config['SWEEP']['VSB']

        nch = self._config.generate_m_dict()
        pch = self._config.generate_m_dict()
        
        dimshape = (len(NSs), len(nch['VGS']), len(nch['VDS']), len(VSBs))
        
        for outvar in self._config['outvars']:
            nch[outvar] = np.zeros(dimshape, order='F')
            pch[outvar] = np.zeros(dimshape, order='F')

        for outvar in self._config['outvars_noise']:
            nch[outvar] = np.zeros(dimshape, order='F')
            pch[outvar] = np.zeros(dimshape, order='F')
        
        # Setup base directory
        os.makedirs("./sweep", exist_ok=True)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            
            for i, stack_count in enumerate(tqdm(NSs, desc="Sweeping NS")):
                self._config._generate_params_file(ns=int(stack_count))                 
                for j, VSB in enumerate(tqdm(VSBs, desc="Sweeping VSB")):
                    self.write_bias_file(sb=VSB)
                    sim_path = f"./sweep/psf_{i}_{j}"
                    os.makedirs(sim_path, exist_ok=True)
                    
                    # Safely write/copy files into the isolated worker directory
                    self._prepare_worker_files(sim_path, VSB)
                    
                    self._simulator.directory = sim_path
                    # Execute from within the isolated directory to prevent worker collisions
                    cp = self._simulator.run('pysweep.scs')

                    futures.append(executor.submit(self.parse_sim, *[sim_path]))
            
            concurrent.futures.wait(futures)

        for f in futures:
            i, j , n_dict, p_dict, nn_dict, pn_dict = f.result()
            for n,p in zip(self._config['n'],self._config['p']):
                params_n = n
                values_n = n_dict[params_n[0]]
                params_p = p
                values_p = p_dict[params_p[0]]
                for m, outvar in enumerate(self._config['outvars']):
                    nch[outvar][i,:,:,j] += np.squeeze(values_n*params_n[2][m])
                    pch[outvar][i,:,:,j] += np.squeeze(values_p*params_p[2][m])

            for n,p in zip(self._config['n_noise'],self._config['p_noise']):
                params_n = n
                values_n = nn_dict[params_n[0]]
                params_p = p
                values_p = pn_dict[params_p[0]]
                for m, outvar in enumerate(self._config['outvars_noise']):
                    nch[outvar][i,:,:,j] += np.squeeze(values_n * params_n[2][m])
                    pch[outvar][i,:,:,j] += np.squeeze(values_p * params_p[2][m])
        self._cleanup()
        
        # Save data to file
        modeln_file_path = f"{self._config['MODEL']['SAVEFILEN']}.pkl"
        modelp_file_path = f"{self._config['MODEL']['SAVEFILEP']}.pkl"
        
        with open(modeln_file_path, 'wb') as f:
            pickle.dump(nch, f)
        with open(modelp_file_path, 'wb') as f:
            pickle.dump(pch, f)
            
        return (modeln_file_path, modelp_file_path)
    
    def parse_sim(self, filepath):
        fileparts = filepath.split("_")
        i = int(fileparts[-2])
        j = int(fileparts[-1])
        
        # Extract Standard AC/DC OP Parameters
        (n_dict, p_dict) = self._extract_sweep_params(filepath, sweep_type="SP")
        
        # Extract Noise Parameters
        (nn_dict, pn_dict) = self._extract_sweep_params(filepath, sweep_type="NOISE")

        return i, j, n_dict, p_dict, nn_dict, pn_dict

    def _prepare_worker_files(self, sim_path, sb):
        """ Isolates the netlist and updates the VSB parameter for thread safety """
        shutil.copy('pysweep.scs', os.path.join(sim_path, 'pysweep.scs'))
        
        # Read the generated params.scs and override the 'sb' variable
        with open('params.scs', 'r') as f:
            param_content = f.read()
            
        param_content = re.sub(r'parameters sb=[\d\.\-]+', f'parameters sb={sb}', param_content)
        
        with open(os.path.join(sim_path, 'params.scs'), 'w') as f:
            f.write(param_content)
    
    def _cleanup(self):            
        pass        
        #try:
        #    shutil.rmtree("./sweep")
        #    if os.path.exists("pysweep.scs"): os.remove("pysweep.scs")
        #    if os.path.exists("params.scs"): os.remove("params.scs")
        #except OSError as e:
        #    print(f"Could not perform cleanup:\nFile - {e.filename}\nError - {e.strerror}")

    def _extract_number_regex(self, string):
        pattern = r'\d+'  
        match = re.search(pattern, string)
        if match:
            return int(match.group())  
        else:
            return None

    def _extract_sweep_params(self, sweep_output_directory, sweep_type="SP"):
        """
        Params  -> list of strings
        size    -> len(VGS) x len(VDS)
        """
        psf_dir = sweep_output_directory
        n_vgs = len(self._config['SWEEP']['VGS'])
        n_vds = len(self._config['SWEEP']['VDS'])
        
        freq = 1.0 # 1 Hz as specified in your netlist
        omega = 2 * np.pi * freq
        
        if sweep_type == "SP":
            params_n = [ k[0] for k in self._config['n'] ]
            params_p = [ k[0] for k in self._config['p'] ]
            
            nmos = {param : np.zeros((n_vgs, n_vds)) for param in params_n}
            pmos = {param : np.zeros((n_vgs, n_vds)) for param in params_p}
            
            # Files across VDS steps
            sp_files_n = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_sp_nmos.sp')), key=self._extract_number_regex)
            sp_files_p = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_sp_pmos.sp')), key=self._extract_number_regex)
            dc_files_n = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_sweepvgs.dc')), key=self._extract_number_regex)
            dc_files_p = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_sweepvgs.dc')), key=self._extract_number_regex)
            for VDS_i, (f_n, f_p, f_dc_n, f_dc_p) in enumerate(zip(sp_files_n, sp_files_p, dc_files_n, dc_files_p)):
                
                # --- NMOS Extraction ---
                psf_n = psf_utils.PSF(f_n)
                psf_dc_n = psf_utils.PSF(f_dc_n)
                
                S_matrix_n = np.zeros((4, 4, n_vgs), dtype=complex)
                for i in range(4):
                    for j in range(4):
                        S_matrix_n[i, j, :] = psf_n.get_signal(f's{i+1}{j+1}').ordinate
                
                Y_n = s2y(S_matrix_n, Z0=50)
                
                nmos['mn:gm'][:, VDS_i]   = np.abs( np.real(Y_n[2, 0, :]) )
                nmos['mn:gds'][:, VDS_i]  = np.abs( np.real(Y_n[2, 2, :]) )
                nmos['mn:gmbs'][:, VDS_i] = np.abs( np.real(Y_n[2, 3, :]) )
                nmos['mn:cgg'][:, VDS_i]  = np.abs( np.imag(Y_n[0, 0, :]) / omega  )
                nmos['mn:cdd'][:, VDS_i]  = np.abs( np.imag(Y_n[2, 2, :]) / omega  )
                nmos['mn:cgd'][:, VDS_i]  = np.abs( -np.imag(Y_n[0, 2, :]) / omega )
                nmos['mn:cgs'][:, VDS_i]  = np.abs( -np.imag(Y_n[0, 1, :]) / omega )
                nmos['mn:cgb'][:, VDS_i]  = np.abs( -np.imag(Y_n[0, 3, :]) / omega )
                nmos['mn:cdg'][:, VDS_i]  = np.abs( -np.imag(Y_n[2, 0, :]) / omega )
                nmos['mn:css'][:, VDS_i]  = np.abs(  np.imag(Y_n[1, 1, :]) / omega )
                nmos['mn:csg'][:, VDS_i]  = np.abs( -np.imag(Y_n[1, 0, :]) / omega )
                
                # Port Current Extraction
                nmos['mn:ids'][:, VDS_i] = np.abs(psf_dc_n.get_signal('PORTND:p').ordinate)
                nmos['mn:igd'][:, VDS_i] = 0 # Static gate current
                nmos['mn:igs'][:, VDS_i] = np.abs(psf_dc_n.get_signal('PORTNG:p').ordinate)
                
                # --- PMOS Extraction ---
                psf_p = psf_utils.PSF(f_p)
                psf_dc_p = psf_utils.PSF(f_dc_p)
                
                S_matrix_p = np.zeros((4, 4, n_vgs), dtype=complex)
                for i in range(4):
                    for j in range(4):
                        S_matrix_p[i, j, :] = psf_p.get_signal(f's{i+1}{j+1}').ordinate
                
                Y_p = s2y(S_matrix_p, Z0=50)
                
                pmos['mp:gm'][:, VDS_i]   = np.abs( np.real(Y_p[2, 0, :]) )
                pmos['mp:gds'][:, VDS_i]  = np.abs( np.real(Y_p[2, 2, :]) )
                pmos['mp:gmbs'][:, VDS_i] = np.abs( np.real(Y_p[2, 3, :]) )
                pmos['mp:cgg'][:, VDS_i]  = np.abs( np.imag(Y_p[0, 0, :]) / omega  )
                pmos['mp:cdd'][:, VDS_i]  = np.abs( np.imag(Y_p[2, 2, :]) / omega  )
                pmos['mp:cgd'][:, VDS_i]  = np.abs( -np.imag(Y_p[0, 2, :]) / omega )
                pmos['mp:cgs'][:, VDS_i]  = np.abs( -np.imag(Y_p[0, 1, :]) / omega )
                pmos['mp:cgb'][:, VDS_i]  = np.abs( -np.imag(Y_p[0, 3, :]) / omega )
                pmos['mp:cdg'][:, VDS_i]  = np.abs( -np.imag(Y_p[2, 0, :]) / omega )
                pmos['mp:css'][:, VDS_i]  = np.abs( np.imag(Y_p[1, 1, :]) / omega  )
                pmos['mp:csg'][:, VDS_i]  = np.abs( -np.imag(Y_p[1, 0, :]) / omega )
                
                pmos['mp:ids'][:, VDS_i] = np.abs(psf_dc_p.get_signal('PORTPD:p').ordinate)
                pmos['mp:igd'][:, VDS_i] = 0
                pmos['mp:igs'][:, VDS_i] = np.abs(psf_dc_p.get_signal('PORTPG:p').ordinate)

        elif sweep_type == "NOISE":
            params_n = [ k[0] for k in self._config['n_noise'] ]
            params_p = [ k[0] for k in self._config['p_noise'] ]
            
            nmos = {param : np.zeros((n_vgs, n_vds)) for param in params_n}
            pmos = {param : np.zeros((n_vgs, n_vds)) for param in params_p}
            
            fl_n = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_noise_nmos_fl.noise')), key=self._extract_number_regex)
            th_n = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_noise_nmos_th.noise')), key=self._extract_number_regex)
            fl_p = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_noise_pmos_fl.noise')), key=self._extract_number_regex)
            th_p = sorted(glob.glob(os.path.join(psf_dir, 'sweep_vds-*_noise_pmos_th.noise')), key=self._extract_number_regex)
            
            for VDS_i, (f_fl_n, f_th_n, f_fl_p, f_th_p) in enumerate(zip(fl_n, th_n, fl_p, th_p)):
                nmos['noise_nmos_fl:out'][:, VDS_i] = psf_utils.PSF(f_fl_n).get_signal('out').ordinate**2
                nmos['noise_nmos_th:out'][:, VDS_i] = psf_utils.PSF(f_th_n).get_signal('out').ordinate**2
                pmos['noise_pmos_fl:out'][:, VDS_i] = psf_utils.PSF(f_fl_p).get_signal('out').ordinate**2
                pmos['noise_pmos_th:out'][:, VDS_i] = psf_utils.PSF(f_th_p).get_signal('out').ordinate**2

        return (nmos, pmos)

    def write_bias_file(self, sb: float = 0.0):
        """ 
        Writes the body bias parameter to VSB.scs. 
        Overwrites the file to ensure only the current bias point exists.
        """
        filename = "vsb.scs"
        try:
            with open(filename, 'w') as f:
                f.write("// VSB.scs - Dynamically generated bias parameter\n")
                f.write("simulator lang=spectre\n")
                f.write(f"parameters sb={sb:.3f}\n")
            # Optional: log for debugging
            # print(f"DEBUG: Successfully updated {filename} with sb={sb:.3f}")
        except IOError as e:
            print(f"Error writing to {filename}: {e}")
