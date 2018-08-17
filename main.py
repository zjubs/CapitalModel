
import busclass2 as bc
import pandas

def service_func():
    print('service func')

if __name__ == '__main__':


    nsims=10
    # set up lobs
    class_data = pandas.read_csv("data/classes.csv")
    lobs = {row['class']: bc.business_class(row['class'], row['grossPrem'],nsims) for index,row in class_data.iterrows()}
    # create loss simulations
    bc.create_dist_losses('attr',"data/attr_params.csv",3)
    bc.create_freq_sev_losses('large',"data/freqsev_params.csv",4)
