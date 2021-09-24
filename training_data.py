import numpy as np
import pylab as pl
import quicklens as ql
import utilities
import random







class TrainingDataGaussBased_Cached():
    #used to create limited size training data to probe how much independent training data is needed
    
    def __init__(self,nx,dx,fnl,fnlmode,nmaps_train=1000,nmaps_valid=1000):
        self.nmaps_train = nmaps_train
        self.nmaps_valid = nmaps_valid
        
        self.nx = nx
        self.dx = dx
        self.fnl = fnl
        self.fnlmode = fnlmode
        
        print ("Generating data set.")
        self.trainingdata_generator = TrainingDataGaussBased(nx,dx,fnl,fnlmode)
        self.lmax = self.trainingdata_generator.lmax
        self.cl_theo = self.trainingdata_generator.cl_theo
        self.cl_theo_ell = self.trainingdata_generator.cl_theo_ell
        self.sample_norm = self.trainingdata_generator.sample_norm
        self.cl_theo_normed = self.trainingdata_generator.cl_theo_normed
        
        self.dataset_train = np.zeros( (self.nmaps_train,self.nx,self.nx) ) 
        for i in range(self.nmaps_train):
            self.dataset_train[i] = self.trainingdata_generator.draw_samples_of_px(1)[0]
        self.dataset_valid = np.zeros( (self.nmaps_valid,self.nx,self.nx) ) 
        self.dataset_valid_logp = np.zeros( (self.nmaps_valid) ) 
        for i in range(self.nmaps_valid):
            sample,logp = self.trainingdata_generator.draw_samples_of_pv(1)
            self.dataset_valid[i] = sample[0]
            self.dataset_valid_logp[i] = logp[0]
        print ("Done.")

    def draw_samples_of_px(self,batch_size): 
        idx = np.random.randint(self.nmaps_train, size=batch_size)
        samples = self.dataset_train[idx,:,:]
        return samples

    def draw_samples_of_pv(self,batch_size):
        idx = np.random.randint(self.nmaps_valid, size=batch_size)
        samples = self.dataset_valid[idx,:,:]
        samples_logp = self.dataset_valid_logp[idx]
        return samples, samples_logp
    
    #this function is needed when fnlmode=True but we need samples from the gaussian
    def draw_samples_of_pv_gauss(self,batch_size):
        #this can be non cached, only needed for fnl norm estimation
        samples, samples_logp = self.trainingdata_generator.draw_samples_of_pv_gauss(batch_size)
        return samples, samples_logp 




class TrainingDataGaussBased():
    #online training data generator that makes maps on the fly
    def __init__(self,nx,dx,fnl,fnlmode):

        self.nx = nx
        self.dx = dx
        self.fnl = fnl
        self.fnlmode = fnlmode
        
        #set ell max automatically
        rffrtmp = ql.maps.rmap(nx,dx).get_rfft()
        ells = np.sort(rffrtmp.get_ell().flatten())
        self.lmax = int(ells[-1]*1.1)
        print ("ell range", ells[0],ells[2],ells[-1],self.lmax)

        #get the power spectrum from camb
        cl_theo_ql = ql.spec.get_camb_lensedcl(lmax=self.lmax)  # cmb theory spectra.
        self.cl_theo = cl_theo_ql.cltt
        self.cl_theo_ell = cl_theo_ql.ls
        self.pix = ql.maps.pix(self.nx, self.dx)
        
        #now determine the sample norm
        self.sample_norm = 1.
        samples_test,_ = self.draw_samples_of_pv_gauss(10000) #NOTE: the non-fnl samples have norm 1 only.
        self.sample_norm = np.std(samples_test)
        self.cl_theo_normed = self.cl_theo/(self.sample_norm*self.sample_norm)



    def draw_samples_of_px(self,batch_size): 
        samples = np.zeros( (batch_size,self.nx,self.nx) )
        for map_id in range(batch_size):
            t_sky,log_p = ql.sims.rfft_density(self.pix, self.cl_theo)

            if not self.fnlmode: #gauss mode
                samples[map_id] = t_sky.get_rmap().map / self.sample_norm
            else: #fnl non gauss mode
                tmap = t_sky.get_rmap().map / self.sample_norm 
                sqmap = tmap*tmap
                sqmean = np.mean(sqmap)
                tmap_fnl = tmap + self.fnl*(sqmap-sqmean)
                samples[map_id] = tmap_fnl 

        return samples

    
    def draw_samples_of_pv(self,batch_size):
        samples = np.zeros( (batch_size,self.nx,self.nx) )
        samples_logp = np.zeros( (batch_size) )

        for map_id in range(batch_size):
            t_sky,log_p = ql.sims.rfft_density(self.pix, self.cl_theo) 

            if not self.fnlmode: #gauss mode
                samples[map_id] = t_sky.get_rmap().map / self.sample_norm
                samples_logp[map_id] = log_p
            else: #fnl non gauss mode
                tmap = t_sky.get_rmap().map / self.sample_norm
                sqmap = tmap*tmap
                sqmean = np.mean(sqmap)
                tmap_fnl = tmap + self.fnl*(sqmap-sqmean)
                samples[map_id] = tmap_fnl 
                samples_logp[map_id] = 0. #as far as i know there is no analytic expression

        return samples, samples_logp
    
    
    #this function is needed when fnlmode=True but we need samples from the gaussian
    def draw_samples_of_pv_gauss(self,batch_size):
        samples = np.zeros( (batch_size,self.nx,self.nx) )
        samples_logp = np.zeros( (batch_size) )

        for map_id in range(batch_size):
            t_sky,log_p = ql.sims.rfft_density(self.pix, self.cl_theo) 
            samples[map_id] = t_sky.get_rmap().map / self.sample_norm
            samples_logp[map_id] = log_p

        return samples, samples_logp 


    
    
class TrainingDataNbodyBased():
    #reads training data from numpy files with patches of simulations
    
    def __init__(self):
        self.lmax = 25000
        self.nx = 128 #128 #64
        self.dx_rad = 125 / 5300. #whole box in rad
        self.dx_degrees = self.dx_rad * utilities.r2d
        self.dx = self.dx_degrees*utilities.d2r / float(self.nx) #one pixel in rad
        self.sample_norm = 1. #enforced below
        self.fnl = 1. #arbitrary normalisation in nbody case

        # Randomly assign training and validation sets
        n_patchfiles = 100
        self.n_trainingfiles = 80
        self.n_validationfiles = n_patchfiles - self.n_trainingfiles

        training_IDs = random.sample(range(n_patchfiles), self.n_trainingfiles)
        validation_IDs = list(range(n_patchfiles))
        for i in range(self.n_trainingfiles):
            validation_IDs.remove(training_IDs[i]) #No validation files are training files

        #print(training_IDs)
        #print(validation_IDs)

        # Read the training sets and validation sets
        datasetfile = []
        fnames_train = []
        for i in range(self.n_trainingfiles):
            fname_train = "../data/patches/patch_"+str(training_IDs[i])+".npy"
            fnames_train.append(fname_train)
            datasetfile.append(np.load(fname_train))

        validationsetfile = []
        fnames_validation = []
        for i in range(self.n_validationfiles):
            fname_validation = "../data/patches/patch_"+str(validation_IDs[i])+".npy"
            fnames_validation.append(fname_validation)
            validationsetfile.append(np.load(fname_validation))

        self.n_dataset = datasetfile[0].shape[0] #Assuming that all patches are the same size
        self.n_validationset = validationsetfile[0].shape[0]
        nsims = datasetfile[0].shape[0]*self.n_trainingfiles

        dataset_full = np.concatenate([np.load(f) for f in fnames_train])
        validationset_full = np.concatenate([np.load(f) for f in fnames_validation])
        print("Dataset shape:", np.shape(dataset_full))
        print("Validationset shape:", np.shape(validationset_full))
        
        dataset = dataset_full[:,:self.nx,:self.nx]
        dataset = dataset/np.std(dataset)
        self.dataset = dataset[:,:,:] - np.mean(dataset)

        validationset = validationset_full[:,:self.nx,:self.nx] 
        validationset = validationset/np.std(validationset)
        self.validationset = validationset[:,:,:] - np.mean(validationset)
        

        print("Data mean:", np.mean(self.dataset[:,:,:]))
        print("Data st. dev.:", np.std(self.dataset[:,:,:]))
        print("Validation mean:", np.mean(self.dataset[:,:,:]))
        print("Validation st. dev.:", np.std(self.dataset[:,:,:]))
        print("Dataset shape:", np.shape(self.dataset))
        print("Validationset shape:", np.shape(self.validationset))

        #get the power spectrum of these samples
        self.lbins      = np.linspace(20, self.lmax, 30) #np.linspace(100, lmax, 30)   
        self.ell_binned = self.lbins[:-1] + 0.5*np.diff(self.lbins)
        cl_avg_true = np.zeros(self.ell_binned.shape[0])
        for map_id in range(self.dataset.shape[0]):
            tmap = self.dataset[map_id,:,:]
            tmap_cfft = ql.maps.rmap(self.nx, self.dx,map=tmap).get_cfft()
            cl = tmap_cfft.get_cl(self.lbins)
            cl_avg_true += cl.cl.real
        self.cl_theo_binned = cl_avg_true/nsims
        self.cl_theo_ell = np.arange(0,self.lmax+1)
        self.pix = ql.maps.pix(self.nx, self.dx)
        self.cl_theo = np.interp( self.cl_theo_ell, self.ell_binned, self.cl_theo_binned, right=0 )
        self.cl_theo_normed = self.cl_theo
        
        
    #Draw random samples from random files
    def draw_samples_of_px(self,batch_size):
        idx = np.random.randint(self.n_trainingfiles*self.n_dataset, size=batch_size)
        samples = self.dataset[idx,:,:] # Random sample in file
        return samples

    
    def draw_samples_of_pv(self,batch_size):
        idx = np.random.randint(self.n_validationfiles*self.n_validationset, size=batch_size)
        vsamples = self.validationset[idx,:,:]
        return vsamples,0
    
    
    def draw_samples_of_pv_gauss(self,batch_size):
        samples = np.zeros( (batch_size,self.nx,self.nx) )
        samples_logp = np.zeros( (batch_size) )
        for map_id in range(batch_size):
            t_sky,log_p = ql.sims.rfft_density(self.pix, self.cl_theo) 
            samples[map_id] = t_sky.get_rmap().map / self.sample_norm
            samples_logp[map_id] = log_p
        return samples, samples_logp 