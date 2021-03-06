"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
def main():
    #Bulid the netowrk
    print ('Building the network')
    net = inet.informationNetwork()
    net.print_information()
    print ('Start running the network')
    net.run_network()
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()
######################################################################
######################################################################
    #Write maximum I(T;Y) in a text file
    print ('Saving maximum I(T;Y)')
    net.write_maxMI()
    #Write I(X;T) and I(T;Y) in a .out text file
    print ('Saving MI curve')
    net.save_MI_plot_data()
######################################################################
######################################################################

if __name__ == '__main__':
    main()
