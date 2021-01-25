import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parse_solenoid_scan_data()

def parse_emittance_data():
    '''read and pasrse data from emittance scans'''

    fname = '2020_summer_cu_inj_emit_data.h5'

    with h5py.File(fname,'r') as f:
        grps = []
        for grp_name in list(f.keys()):
            if 'OTR2' in grp_name:
                grps += [f[grp_name]]
                try:
                    sol_val = f[grp_name]['pvdata'].attrs['SOLN:IN20:121:BDES']
                    print(f'{grp_name} {sol_val}')
                except KeyError:
                    print(f'{grp_name}')


def get_solenoid_scan_data():
    fname = 'measurements/2020_summer_cu_inj_solscan_data.h5'

    with h5py.File(fname,'r') as f:
        grps = []
        for grp_name in list(f.keys()):
            if 'OTR2' in grp_name:
                grps += [f[grp_name]]
                #print(grp_name)


        #create dataframe with relevant data
        data = []
        
        for i in range(len(grps)):
            g = grps[i]
            for name, val in g['beam_data'].items():
                #combine setpoints
                sol_info = dict(val.attrs)
                pvdata = dict(g['pvdata'].attrs)
                pvdata.update(sol_info)

                #add screen data
                n = ['stats_XRMS','stats_YRMS']
                rms_sizes = {ele:val['sample0']['Gaussian'][ele][()] for ele in n}
                
                pvdata.update(rms_sizes)

                pvdata.update({'scan_number':i})
                data += [pvdata]
                
        df = pd.DataFrame(data)

    #remove BDES column and rename BCTRL
    df = df.drop(labels = 'SOLN:IN20:121:BDES', axis = 1)
    df = df.rename(columns = {'SOLN:IN20:121:BCTRL':'SOLN:IN20:121:BDES'})
        

    return df

if __name__ == '__main__':
    main()
    plt.show()
