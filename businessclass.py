from pandas import DataFrame
import pandas
from scipy.stats import genpareto,lognorm,poisson

class business_class():
    def __init__(self,name,grossprem,nsims):
        self.name = name
        self.nsims = nsims
        self.grossprem = grossprem
        self.sim_nums = list(range(1,self.nsims+1))
        self.grosslosses=DataFrame(index = self.sim_nums)
        self.netlosses=DataFrame(index = self.sim_nums)
        
        self.f = {
            'lognorm': lognorm.rvs,
            'poisson': poisson.rvs,
            'genpareto': genpareto.rvs
            }
            
    
    def addlosses(self,loss_df,loss_name):
        #append array of losses to existing losses
        self.grosslosses[loss_name] = loss_df
    
    def sim_distr_losses(self, distr,loss_name="attr",params=[]):
        # simulate attritional losses
        # maybe better called distribution
        # need to feed in params and dist choice
        distr_loss = DataFrame(self.f[distr](*params,size = self.nsims), index = self.sim_nums)
        self.addlosses(distr_loss,loss_name)
    
    def sim_freqsev_losses(self, freq_distr, sev_distr,loss_name="large",freq_params=[], sev_params=[]):
        # maybe better called freq_sev
        # need to feed in params and dist choice
        freq_sev_losses = DataFrame(self.f[freq_distr](*freq_params,size = self.nsims),columns =["freq"], index = self.sim_nums)
        #large_loss = DataFrame(self.f[sev_distr](*sev_params,size =self.nsims),columns =["sev"], index = self.sim_nums)
        freq_sev_losses['sev'] = [ self.f[sev_distr](*sev_params,size = freq) for freq in freq_sev_losses['freq']]
        freq_sev_losses['agg'] = [ sum(losses) for losses in freq_sev_losses['sev']]
        self.addlosses(freq_sev_losses['agg'],loss_name)
    
    def apply_dependency(self):
        print ("a")  #reorder column in grosslosses, will need to reorder indices

################################

def read_params(paramsfile_path,info_end_col):
    param_data = pandas.read_csv(paramsfile_path)
    param_data['params'] = param_data.iloc[:,info_end_col:].values.tolist()
    #remove nan values
    for index,row in param_data.iterrows():
        clean_list = [x for x in row['params'] if str(x) != 'nan']
        param_data.set_value(index,'params', clean_list)
    return param_data

def create_dist_losses(loss_name,paramsfile_path,info_end_col):
    """
    takes a file as an input and the index of the last column before parameters
    and uses this information to read in data and simulate losses for the business_class
    objects
    """
    loss_params=read_params(paramsfile_path,info_end_col)
    for index,row in loss_params.iterrows():
        lobs[row['class']].sim_distr_losses(loss_params.loc[index,'dist'],loss_name,loss_params.loc[index,'params'])
        

def create_freq_sev_losses(loss_name, paramsfile_path,info_end_col):
    """
    takes a file as an input and the index of the last column before parameters
    and uses this information to read in data and simulate losses for the business_class
    objects. There is probably a more concise way to do this!!!!
    """
    x=read_params("data/freqsev_params.csv",4)
    freq_rows = x[x['freq_sev'] == 'freq']
    sev_rows = x[x['freq_sev'] == 'sev']
    classes = freq_rows['class']
    for item in classes:
        freq_class_inputs  = freq_rows[freq_rows['class'] == item].iloc[0]
        sev_class_inputs  = sev_rows[sev_rows['class'] == item].iloc[0]
        lobs[item].sim_freqsev_losses(freq_class_inputs['dist'],sev_class_inputs['dist'],
                                  loss_name, freq_class_inputs['params'], sev_class_inputs['params'])

nsims=10
# set up lobs
class_data = pandas.read_csv("data/classes.csv")
lobs = {row['class']: business_class(row['class'], row['grossPrem'],nsims) for index,row in class_data.iterrows()}


create_dist_losses('attr',"data/attr_params.csv",3)
create_freq_sev_losses('large',"data/freqsev_params.csv",4)


# should be able to use multi index and pass in a list of column names
# suggested index: YOA,loss_type:attr/large/cat, exposure_type: earned/unearned/new, peril: cat only
# should be flexible enough to work with less indices if necessary

"""
#create attr and large losses
# read in attr params
attr=read_params("data/attr_params.csv",3)
large=read_params("data/freqsev_params.csv",4)

# create attr losses
for index,row in attr.iterrows():
    lobs[row['class']].sim_distr_losses(attr.loc[index,'dist'],attr.loc[index,'type'],attr.loc[index,'params'])



# create large losses
freq_rows = large[large['freq_sev'] == 'freq']
sev_rows = large[large['freq_sev'] == 'sev']
classes = freq_rows['class']
for item in classes:
    freq_class_inputs  = freq_rows[freq_rows['class'] == item].iloc[0]
    sev_class_inputs  = sev_rows[sev_rows['class'] == item].iloc[0]
    lobs[item].sim_freqsev_losses(freq_class_inputs['dist'],sev_class_inputs['dist'],
                                  'large', freq_class_inputs['params'], sev_class_inputs['params'])




# sim losses for individual lob
#lobs['Property'].sim_distr_losses(attr.loc[0,'dist'],attr.loc[0,'type'],attr.loc[0,'params']) 
# lobs['Property'].sim_freqsev_losses('poisson','genpareto','large', [5], [0.5,100,50])

 
    #########################################
   # turning strings into functions     
    functions = {
    'sum': sum,
    'mean': lambda v: sum(v) / len(v),
}

then look up functions from that dictionary instead:

functions['sum'](range(1, 11))
 #####################################################   
     
from pandas import DataFrame
from scipy.stats import genpareto,lognorm,poisson 
temp = DataFrame(genpareto.rvs(0.5,100,50,size =10),columns =["blah"])
temp2 = DataFrame(genpareto.rvs(0.5,100,50,size =10),columns =["blah2"])
temp = temp.combine_first(temp2)
temp3 = DataFrame(lognorm.rvs(s=5,scale = 100,size =10),columns =["blah2"])

"""
