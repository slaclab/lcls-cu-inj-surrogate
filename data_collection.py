import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import read_exp

    
def get_sim_data():
    X_df = pd.read_pickle('frontend/X.p')
    Y_df = pd.read_pickle('frontend/Y.p')

    return X_df, Y_df
    
def get_exp_data(single_scan = False):
    #import mapping between Impact and real parameters
    mapping = pd.read_csv('pv_mapping/cu_inj_impact.csv')
    mapping.index = mapping['device_pv_name']

    #import data frame from measurements
    if single_scan:
        measurements0 = read_exp.get_single_solenoid_scan()
    else:
        measurements0 = read_exp.parse_solenoid_scan_data()

    #remove units to get scaling
    measurements = measurements0[[col_name for col_name in measurements0 if not '.EGU' in col_name]]

    #remove base solenoid strength
    measurements = measurements.drop(labels = ['SOLN:IN20:121:BDES'],axis=1)
    #print(measurements)

    scaled_measurements_raw = {}
    
    #transform measurement pv's into simulation pv's w/name and add to dataframe
    for col_name, col_values in measurements.items():
        if col_name == 'SOLN:IN20:121:BCTRL':
            col_name = 'SOLN:IN20:121:BDES'

        if not 'stats' in col_name: 
            #get scaling for col_name
            scale = mapping.loc[col_name]['impact_factor']
            impact_name = mapping.loc[col_name]['impact_name']

            #print(scale)
            #print(f'{col_name} : {impact_name} : {scale}')

            #scale each data column and rename the column for new dataframe
            if col_name == 'IRIS:LR20:130:MOTR_ANGLE':
                scaled_measurements_raw[impact_name] = convert_iris_diameter(np.asfarray(col_values))
            else:
                scaled_measurements_raw[impact_name] = scale * np.asfarray(col_values)

        else:
            scaled_measurements_raw[col_name] = 1e-6 * np.asfarray(col_values)
            

    #add laser pulse length
    scaled_measurements_raw['distgen:t_dist:length:value'] = 4.0
    scaled_measurements = pd.DataFrame(scaled_measurements_raw)
    #print(scaled_measurements)
    return scaled_measurements

def convert_iris_diameter(X):
    fit_data = np.loadtxt('pv_mapping/iris_diameter_mapping.md', skiprows = 4, max_rows = 10)

    #linear fit to data
    z = np.polyfit(*fit_data.T[::-1],1)
    p = np.poly1d(z)
    #plt.plot(*fit_data.T[::-1])

    return p(X) / 3.0


if __name__ == '__main__':    
    main()
    plt.show()
