from pandas import DataFrame
import pandas
from scipy.stats import genpareto,lognorm,poisson, uniform
import numpy as np

class business_class():
    def __init__(self,name,grossprem,nsims):
        self.name = name
        self.nsims = nsims
        self.grossprem = grossprem
        self.sim_nums = list(range(1,self.nsims+1))
        self.grosslosses=np.array([]).reshape(0,self.nsims)
        self.grosslosslabels= []
        self.netlosses=np.array([]).reshape(0,self.nsims)
        
        self.loss_keys= ['YOA', 'AY', 'loss_type', 'exposure_type', 'loss_name']
        #YOA, AY, loss_type:attr/large/cat, exposure_type: earned/unearned/new, name/peril
        #z = dict.fromkeys(a)
        
        self.f = {
            'lognorm': lognorm.rvs,
            'poisson': poisson.rvs,
            'genpareto': genpareto.rvs,
            'uniform': uniform.rvs
            }
    
    def addlosses(self,loss_df,loss_labels):
        """
        appends losses ot grossloss
        appends dictionary of labels as a list to list of grosslosslabels
        """
        
        print(self.grosslosses)
        print(loss_df)
        self.grosslosses = np.concatenate((self.grosslosses,loss_df),axis=0)
        self.grosslosslabels.append([loss_labels])
    
    def sim_distr_losses(self, distr,loss_labels,params=[]):
        # simulate attritional losses
        # maybe better called distribution
        # need to feed in params and dist choice
        print(distr)
        print(loss_labels)
        print(params)
        distr_loss = np.array(self.f[distr](*params,size = self.nsims)).reshape(1,self.nsims)
        distr_loss.sort() # sort ascending
        print('here')
        self.addlosses(distr_loss,loss_labels)
    
    def sim_freqsev_losses(self, freq_distr, sev_distr,loss_labels,freq_params=[], sev_params=[]):
        # maybe better called freq_sev
        # need to feed in params and dist choice
        freq_sev_losses = np.array(self.f[freq_distr](*freq_params,size = self.nsims))
        #large_loss = np.array(self.f[sev_distr](*sev_params,size =self.nsims))
        freq_sev_losses_sev = np.array([ self.f[sev_distr](*sev_params,size = freq) for freq in freq_sev_losses]).reshape(1,self.nsims)
        print(freq_sev_losses_sev)        
        #order ascending
        self.temp = freq_sev_losses_sev
        a =[sum(i) for i in freq_sev_losses_sev[0]]
        print(a)
        freq_sev_losses_sev = [freq_sev_losses_sev[0,np.argsort(a)]]
        print(freq_sev_losses_sev)
        print(loss_labels)
        self.temp2 = freq_sev_losses_sev
        #freq_sev_losses_agg = [ sum(losses) for losses in freq_sev_losses_sev]
        self.addlosses(freq_sev_losses_sev,loss_labels)
    
    def apply_dependency(self):
        print ("a")  #reorder column in grosslosses, will need to reorder indices
        
    def apply_earning_pattern(self):
        #apply earning pattern to loss to allocate to AYs
        print("a")


################################

def read_params(paramsfile_path,info_end_col):
    """
    takes a file as an input and the index of the last column before parameters
    and uses this information to read in data and simulate losses for the business_class
    objects
    """
    param_data = pandas.read_csv(paramsfile_path)
    param_data['params'] = param_data.iloc[:,info_end_col:].values.tolist()
    #remove nan values
    for index,row in param_data.iterrows():
        clean_list = [x for x in row['params'] if str(x) != 'nan']
        param_data.set_value(index,'params', clean_list)
    return param_data
    #param_data['labels'] = 

# loss indices shoul also be read into lob when initialising
#input data must have samw order as loss indices
#loss_indices = ['YOA', 'AY', 'loss_type', 'exposure_type', 'loss_name']
#a = read_params("data/attr_params.csv",6)
#dict(zip(loss_indices,[1,2,3,4,5]))


def create_dist_losses(paramsfile_path,info_end_col, loss_indices):
    """
    takes a file as an input and the index of the last column before parameters
    and uses this information to read in data and simulate losses for the business_class
    objects
    """
    loss_params=read_params(paramsfile_path,info_end_col)
    for index,row in loss_params.iterrows():
        lobs[row['class']].sim_distr_losses(loss_params.loc[index,'dist'],
        loss_params.loc[index,loss_indices[0]:loss_indices[-1]].to_dict(),
        loss_params.loc[index,'params']) #parans
        

def create_freq_sev_losses( paramsfile_path,info_end_col,loss_indices):
    """
    takes a file as an input and the index of the last column before parameters
    and uses this information to read in data and simulate losses for the business_class
    objects. There is probably a more concise way to do this!!!!
    """
    x=read_params("data/freqsev_params.csv",info_end_col)
    freq_rows = x[x['freq_sev'] == 'freq']
    sev_rows = x[x['freq_sev'] == 'sev']
    classes = freq_rows['class']
    for item in classes:
        freq_class_inputs  = freq_rows[freq_rows['class'] == item].iloc[0]
        sev_class_inputs  = sev_rows[sev_rows['class'] == item].iloc[0]
        loss_label = freq_class_inputs.loc[loss_indices[0]:loss_indices[-1]].to_dict()
        lobs[item].sim_freqsev_losses(freq_class_inputs['dist'],sev_class_inputs['dist'],
                                  loss_label, freq_class_inputs['params'], sev_class_inputs['params'])

nsims=10
loss_indices = ['YOA', 'AY', 'loss_type', 'exposure_type', 'loss_name']
# set up lobs

class_data = pandas.read_csv("data/classes.csv")
lobs = {row['class']: business_class(row['class'], row['grossPrem'],nsims) for index,row in class_data.iterrows()}


create_dist_losses("data/attr_params.csv",7,loss_indices)
create_freq_sev_losses("data/freqsev_params.csv",8, loss_indices)

########################

temp  = lobs['Property'].grosslosses[1]
a =[sum(i) for i in temp]
temp[np.argsort(a)]

temp0= uniform.rvs(0,1,size =10)
vals = uniform.rvs(0,1,size =10)
order_index = np.argsort(temp0) 
z = dict(zip(vals,order_index))



##############
lobs['Property'].sim_freqsev_losses('poisson','genpareto','test_loss', [3], [0.5,100,50])
a_freq_sev_losses = np.array(poisson.rvs(*[3],size=10))
a_freq_sev_losses_sev = np.array([ genpareto.rvs(*[0.5,100,50],size = freq) for freq in freq_sev_losses]).reshape(1,10)
self.freqsev = freq_sev_losses_sev
freq_sev_losses_agg = [ sum(losses) for losses in freq_sev_losses_sev]
self.addlosses(freq_sev_losses_agg,loss_name)
##############
business_class('trop', 100,nsims)
lobs['Property'].sim_distr_losses(attr.loc[0,'dist'],attr.loc[0,'type'],attr.loc[0,'params']) 

attr = read_params("data/attr_params.csv",3)

lobs['Property'].grosslosses
lobs['Property'].test
dtype = [('blah'), 'float']




