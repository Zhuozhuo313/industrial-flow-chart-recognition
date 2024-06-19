# 结尾带*的key仅用作识别，实际上不存在该类optype
operator_keyword_dict = {'AVG':['AVG'], 'VOTER':['VOTER'], '|X1-X2|':['LIM','|X1-X2|'],
                         'T1_0s':['T1','0s'], 'LIM_2':['LIM','X','D'], 'MAX':['MAX'], 'FX':['FX'], 'MIN':['MIN'],
                         '0s_T2':['0s','T2'], 'B_Input':['B_Input'], 'A_Input':['A_Input'], '3/4':['3/4'], 'Qlnv':['Qlnv'],
                         '2/3':['2/3'], '2/4':['2/4'], 'T1_T2':['T1','T2'],
                         'dx/dt':['dx','dt'], '&':['&'], '1/1+Ts':['1+Ts'], 'PID':['PID'], '1+T1s/1+T2s':['1+T1s','1+T2s'],
                         'Tmax':['Tmax'], 'Graph':['P_SLO','N_SLO']}

operator_template_dict = {'root':['root',0], '>=1':["mt1",0], 'LXU':['LXU',0], 'LXU*':['LXU2',0], 'SVR':['SVR',0], '1':['1',2], 'SUM':['SUM',0],
                          'C':['C',0], 'C*':['C2',0], 'LMT':['LMT',0], 'LIM_1':['LIM_1',0], 'LIM_2':['LIM_2',0], '&': ['and',0], '&*': ['and2',0],
                          't':['t',0], 'abs':['abs',0], 'up':['up',0],'>=1_2':['mt1_2',0], 'X1/X2':['X1_X2',0], 'inv':['inv',1],
                          'TSY':['TSY',0], 'dx/dt':['dx_dt',0], 'X':['X',1], 'SRV':['SRV',0]}

start_special_distance = {'PID':15, 'LXU':15, 'LIM_1':15, 'LIM_2':15, 'dx/dt':15, '|X1-X2|':21, 'SRV':19, 'SVR':19, 'VOTER':19, '1/1+Ts':14}