from Util import *
import numpy as np
import random

class Mol:
	""" Provides a general purpose molecule"""
	def __init__(self, atoms_ =  None, coords_ = None, mbe_order_ =5):
		if (atoms_!=None):
			self.atoms = atoms_
		else: 
			self.atoms = np.zeros(1,dtype=np.uint8)
		if (coords_!=None):
			self.coords = coords_
		else:
			self.coords=np.zeros(shape=(1,1),dtype=np.float)
		self.properties = {"MW":0}
		self.name=None
		#things below here are sometimes populated if it is useful.
		self.PESSamples = [] # a list of tuples (atom, new coordinates, energy) for storage.
		self.ecoords = None # equilibrium coordinates.
		self.DistMatrix = None # a list of equilbrium distances, for GO-models.
		self.mbe_order = mbe_order_
		self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
		self.mbe_frags_deri=dict()
		self.mbe_permute_frags=dict() # list of all the permuted frags
		self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
		self.energy=None
		self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
		self.mbe_deri =None
		self.nn_energy=None
		return


	def Reset_Frags(self):
                self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
                self.mbe_frags_deri=dict()
                self.mbe_permute_frags=dict() # list of all the permuted frags
                self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
                self.energy=None
                self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
                self.mbe_deri =None
                self.nn_energy=None
                return


	def AtomName(self, i):
             	return atoi.keys()[atoi.values().index(self.atoms[i])]

	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names

	def IsIsomer(self,other):
		return np.array_equals(np.sort(self.atoms),np.sort(other.atoms))
			
	def NAtoms(self):
		return self.atoms.shape[0]

	def AtomsWithin(self,rad, pt):
		# Returns indices of atoms within radius of point.
		dists = map(lambda x: np.linalg.norm(x-pt),self.coords)
		return [i for i in range(self.NAtoms()) if dists[i]<rad]

	def NumOfAtomsE(self, at):
		return sum( [1 if e==at else 0 for e in self.atoms ] )

	def Rotate(self,axis,ang):
		rm=RotationMatrix(axis,ang)
		crds=np.copy(self.coords)
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(rm,crds[i])

	def MoveToCenter(self):
		first_atom = (self.coords[0]).copy()
		for i in range (0, self.NAtoms()):
			self.coords[i] = self.coords[i] - first_atom

	def AtomsWithin(self, SensRadius, coord):
		''' Returns atoms within the sensory radius in sorted order. '''
		satoms=np.arange(0,self.NAtoms())
		diffs= self.coords-coord
		dists= np.power(np.sum(diffs*diffs,axis=1),0.5)
		idx=np.argsort(dists)
		mxidx = len(idx)
		for i in range(self.NAtoms()):
			if (dists[idx[i]] >= SensRadius):
				mxidx=i
				break
		return idx[:mxidx]

	def WriteXYZfile(self, fpath=".", fname="mol"):
		if (os.path.isfile(fpath+"/"+fname+".xyz")):
			f = open(fpath+"/"+fname+".xyz","a")
		else:
			f = open(fpath+"/"+fname+".xyz","w")
		natom = self.atoms.shape[0]
		f.write(str(natom)+"\n\n")
		for i in range (0, natom):
			atom_name =  atoi.keys()[atoi.values().index(self.atoms[i])]
			f.write(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2])+"\n")	
		f.write("\n\n")	
		f.close()

	def Distort(self,seed=0,disp=0.35):
		''' Randomly distort my coords, but save eq. coords first '''
		self.BuildDistanceMatrix()
		random.seed(seed)
		for i in range (0, self.atoms.shape[0]):
			for j in range (0, 3):
				self.coords[i,j] = self.coords[i,j] + disp*random.uniform(-1, 1)

	def AtomTypes(self):
		return np.unique(self.atoms)

	def ReadGDB9(self,path,mbe_order=3):
		try:
			f=open(path,"r")
			lines=f.readlines()
			natoms=int(lines[0])
			self.mbe_order=mbe_order
			self.atoms.resize((natoms))
			self.coords.resize((natoms,3))
			for i in range(natoms):
				line = lines[i+2].split()
				self.atoms[i]=AtomicNumber(line[0])
				try:
					self.coords[i,0]=float(line[1])
				except:
					self.coords[i,0]=scitodeci(line[1])
				try:
					self.coords[i,1]=float(line[2])
				except:
					self.coords[i,1]=scitodeci(line[2])
				try:
					self.coords[i,2]=float(line[3])
				except:
					self.coords[i,2]=scitodeci(line[3])
			f.close()
		except Exception as Ex:
			print "Read Failed.", Ex
			raise Ex
		return

	def FromXYZString(self,string):
		lines = string.split("\n")
		natoms=int(lines[1])
		self.atoms.resize((natoms))
		self.coords.resize((natoms,3))
		for i in range(natoms):
			line = lines[i+3].split()
			self.atoms[i]=AtomicNumber(line[0])
			try:
				self.coords[i,0]=float(line[1])
			except:
				self.coords[i,0]=scitodeci(line[1])
			try:
				self.coords[i,1]=float(line[2])
			except:
				self.coords[i,1]=scitodeci(line[2])
			try:
				self.coords[i,2]=float(line[3])
			except:
				self.coords[i,2]=scitodeci(line[3])
		return

	def NEle(self):
		return np.sum(self.atoms)

	def XYZtoGridIndex(self, xyz, ngrids = 250,padding = 2.0):
		Max = (self.coords).max() + padding
                Min = (self.coords).min() - padding
		binsize = (Max-Min)/float(ngrids-1)
		x_index = math.floor((xyz[0]-Min)/binsize)
		y_index = math.floor((xyz[1]-Min)/binsize)
		z_index = math.floor((xyz[2]-Min)/binsize)
#		index=int(x_index+y_index*ngrids+z_index*ngrids*ngrids)
		return x_index, y_index, z_index

	def MolDots(self, ngrids = 250 , padding =2.0, width = 2):
		grids = self.MolGrids()
		for i in range (0, self.atoms.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(self.coords[i])
			for m in range (-width, width):	
				for n in range (-width, width):
					for k in range (-width, width):
						index = (x_index)+m + (y_index+n)*ngrids + (z_index+k)*ngrids*ngrids
						grids[index] = atoc[self.atoms[i]]
		return grids

	def Center(self):
		''' Returns the center of atom'''
		return np.average(self.coords,axis=0)

	def rms(self, m):
		err  = 0.0
		for i in range (0, (self.coords).shape[0]):
			err += (np.sum((m.coords[i] - self.coords[i])**2))**0.5
		return err/float((self.coords).shape[0])

	def MolGrids(self, ngrids = 250):
		grids = np.zeros((ngrids, ngrids, ngrids), dtype=np.uint8)
		grids = grids.reshape(ngrids**3)   #kind of ugly, but lets keep it for now
		return grids

	def SpanningGrid(self,num=250,pad=4.):
		''' Returns a regular grid the molecule fits into '''
		xmin=np.min(self.coords[:,0])-pad
		xmax=np.max(self.coords[:,0])+pad
		ymin=np.min(self.coords[:,1])-pad
		ymax=np.max(self.coords[:,1])+pad
		zmin=np.min(self.coords[:,2])-pad
		zmax=np.max(self.coords[:,2])+pad
		grids = np.mgrid[xmin:xmax:num*1j, ymin:ymax:num*1j, zmin:zmax:num*1j]
		grids = grids.transpose()
		grids = grids.reshape((grids.shape[0]*grids.shape[1]*grids.shape[2], grids.shape[3]))
		return grids, (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

	def AddPointstoMolDots(self, grids, points, value, ngrids =250):  # points: x,y,z,prob    prob is in (0,1)
		points = points.reshape((-1,3))  # flat it
		value = value.reshape(points.shape[0]) # flat it
		value = value/value.max()
		for i in range (0, points.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(points[i])
			index = x_index + y_index*ngrids + z_index*ngrids*ngrids
			if grids[index] <  int(value[i]*250):
				grids[index] = int(value[i]*250)
		return grids
	
	def GridstoRaw(self, grids, ngrids=250, save_name="mol", save_path ="./densities/"):
		if (save_name=="mol" and self.name != None):
			save_name=self.name
		grids = np.array(grids, dtype=np.uint8)
		print "Saving density to:",save_path+save_name+".raw"
		f = open(save_path+save_name+".raw", "wb")
		f.write(bytes(np.array([ngrids,ngrids,ngrids],dtype=np.uint8).tostring())+bytes(grids.tostring()))
		f.close()


	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0):
		for i in range (1, self.mbe_order+1):
			self.Generate_MBE_term(i, atom_group, cutoff, center_atom)
		return  0

	def Generate_MBE_term(self, order,  atom_group=1, cutoff=10, center_atom=0):
		if order in self.mbe_frags.keys():
                        print ("MBE order", order, "already generated..skipping..")
			return 0
		if (self.coords).shape[0]%atom_group!=0:
			raise Exception("check number of group size")
		else:
			ngroup = (self.coords).shape[0]/atom_group

		xyz=((self.coords).reshape((ngroup, atom_group, -1))).copy()     # cluster/molecule needs to be arranged with molecule/sub_molecule
		ele=((self.atoms).reshape((ngroup, atom_group))).copy()
		mbe_terms=[]
		mbe_terms_num=0
		mbe_dist=[]
		atomlist=list(range(0,ngroup))		

		if order < 1 :
			raise Exception("MBE Order Should be Positive")
		else:	
			time_log = time.time()
			print ("generating the combinations..")
			combinations=list(itertools.combinations(atomlist,order))
			print ("finished..takes", time_log-time.time(),"second")
		time_now=time.time()
		flag = np.zeros(1)
		max_case = 5000   #  set max cases for debug  
		for i in range (0, len(combinations)):
			term = list(combinations[i])
			pairs=list(itertools.combinations(term, 2))	
			saveindex=[]
			dist = [10000000]*len(pairs)
#			flag = 1
#			for j in range (0, len(pairs)):
#				m=pairs[j][0]
#				n=pairs[j][1]
#				#dist[j] = np.linalg.norm(xyz[m]-xyz[n])
#				dist[j]=((xyz[m][center_atom][0]-xyz[n][center_atom][0])**2+(xyz[m][center_atom][1]-xyz[n][center_atom][1])**2+(xyz[m][center_atom][2]-xyz[n][center_atom][2])**2)**0.5
#				if dist[j] > cutoff:
#					flag = 0
#					break
#			if flag == 1:
			flag[0]=1
			npairs=len(pairs)
			code="""
			for (int j=0; j<npairs; j++) {
				int m = pairs[j][0];
				int n = pairs[j][1];
				dist[j] = sqrt(pow(xyz[m*atom_group*3+center_atom*3+0]-xyz[n*atom_group*3+center_atom*3+0],2)+pow(xyz[m*atom_group*3+center_atom*3+1]-xyz[n*atom_group*3+center_atom*3+1],2)+pow(xyz[m*atom_group*3+center_atom*3+2]-xyz[n*atom_group*3+center_atom*3+2],2));
				if (float(dist[j]) > cutoff) {
					flag[0] = 0;
					break;
				}
			}
			
			"""
			res = inline(code, ['pairs','npairs','center_atom','dist','xyz','flag','cutoff','atom_group'],headers=['<math.h>','<iostream>'], compiler='gcc')
			if flag[0]==1:  # end of weave
				if mbe_terms_num%100==0:
					#print mbe_terms_num, time.time()-time_now
					time_now= time.time()
				mbe_terms_num += 1
				mbe_terms.append(term)
				mbe_dist.append(dist)
				if mbe_terms_num >=  max_case:   # just for generating training case
                                        break;

		mbe_frags = []
		for i in range (0, mbe_terms_num):
			tmp_atom = np.zeros(order*atom_group) 
			tmp_coord = np.zeros((order*atom_group, 3))  
			for j in range (0, order):
				tmp_atom[atom_group*j:atom_group*(j+1)] = ele[mbe_terms[i][j]]
				tmp_coord[atom_group*j:atom_group*(j+1)] = xyz[mbe_terms[i][j]]
			tmp_mol = Frag(tmp_atom, tmp_coord, mbe_terms[i], mbe_dist[i], atom_group)
			mbe_frags.append(tmp_mol)
		self.mbe_frags[order]=mbe_frags
		print "generated {:10d} terms for order {:d}".format(len(mbe_frags), order)
		return mbe_frags

	def Calculate_Frag_Energy(self, order):
		if order in self.mbe_frags_energy.keys():
			print ("MBE order", order, "already calculated..skipping..")
			return 0
		mbe_frags_energy = 0.0

		fragnum=0
		time_log=time.time()
		max_case = 5000   # set max_case for debug
		print "length of order ", order, ":",len(self.mbe_frags[order])
		for frag in self.mbe_frags[order][0:max_case]:  # just for generating the training set..
			fragnum +=1;
			print "doing the ",fragnum
			frag.PySCF_Frag_MBE_Energy_All()
			frag.Set_Frag_MBE_Energy()
			mbe_frags_energy += frag.frag_mbe_energy
			print "Finished, spent ", time.time()-time_log," seconds"
			time_log = time.time()
		self.mbe_frags_energy[order] = mbe_frags_energy
		return 0
	
	def Calculate_All_Frag_Energy(self):  # we ignore the 1st order for He here
		for i in range (2, self.mbe_order+1):
			print "calculating for MBE order", i
			self.Calculate_Frag_Energy(i)
		print "mbe_frags_energy", self.mbe_frags_energy
		return 0	

	def Set_MBE_Energy(self):
		for i in range (1, self.mbe_order+1): 
			self.mbe_energy[i] = 0.0
			for j in range (1, i+1):
				self.mbe_energy[i] += self.mbe_frags_energy[j]
		return 0.0


	def MBE(self,  atom_group=1, cutoff=10, center_atom=0):
		self.Generate_All_MBE_term(atom_group, cutoff, center_atom)
                self.Calculate_All_Frag_Energy()
                self.Set_MBE_Energy()
		print self.mbe_frags_energy
		return 0


	
	def PySCF_Energy(self):
		mol = gto.Mole()
                pyscfatomstring=""
                for j in range(len(self.atoms)):
			s = self.coords[j]
                	pyscfatomstring=pyscfatomstring+str(self.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(self.atoms)-1 else "")
                mol.atom = pyscfatomstring
                mol.basis = 'cc-pvqz'
                mol.verbose = 0
                try:
			mol.build()
                        mf=scf.RHF(mol)
                        hf_en = mf.kernel()
                        mp2 = mp.MP2(mf)
                        mp2_en = mp2.kernel()
                        en = hf_en + mp2_en[0]
			self.energy = en
			return en
                except Exception as Ex:
                        print "PYSCF Calculation error... :",Ex
                        print "Mol.atom:", mol.atom
                        print "Pyscf string:", pyscfatomstring
                        return 0.0
                        #raise Ex
                return 0.0
			

	def Get_Permute_Frags(self):
		self.mbe_permute_frags=dict()
		for order in self.mbe_frags.keys():
			self.mbe_permute_frags[order]=list()
			for frags in self.mbe_frags[order]:
				self.mbe_permute_frags[order] += frags.Permute_Frag()
			print "length of permuted frags:", len(self.mbe_permute_frags[order]),"order:", order
		return 


	def Set_Frag_Force_with_Order(self, cm_deri, nn_deri, order):
		self.mbe_frags_deri[order]=np.zeros((self.NAtoms(),3))
		atom_group = self.mbe_frags[order][0].atom_group  # get the number of  atoms per group by looking at the frags.
		for i in range (0, len(self.mbe_frags[order])):
			deri = self.mbe_frags[order][i].Frag_Force(cm_deri[i], nn_deri[i])
			index_list = self.mbe_frags[order][i].index
			for j in range (0,  len(index_list)):
				self.mbe_frags_deri[order][index_list[j]*atom_group:(index_list[j]+1)*atom_group] += deri[j]
		#print "derviative from order ",order, self.mbe_frags_deri[order]
		return 

	def Set_MBE_Force(self):
		self.mbe_deri = np.zeros((self.NAtoms(), 3))
		for order in range (2, self.mbe_order+1): # we ignore the 1st order term since we are dealing with helium
			if order in self.mbe_frags_deri.keys():
				self.mbe_deri += self.mbe_frags_deri[order]
		return self.mbe_deri
	

	
class Frag(Mol):
        """ Provides a MBE frag of  general purpose molecule"""
        def __init__(self, atoms_ =  None, coords_ = None, index_=None, dist_=None, atom_group_=1):
		Mol.__init__(self, atoms_, coords_)
		self.atom_group = atom_group_
		self.FragOrder = self.coords.shape[0]/self.atom_group
		if (index_!=None):
			self.index = index_
		else:
			self.index = None
		if (dist_!=None):
			self.dist = dist_
		else:
			self.dist = None
		self.frag_mbe_energies=dict()
		self.frag_mbe_energy = None
		self.permute_index = range (0, self.FragOrder)	
                return
			

	def PySCF_Frag_MBE_Energy(self,order):   # calculate the MBE of order N of each frag 
		inner_index = range(0, self.FragOrder) 
		real_frag_index=list(itertools.combinations(inner_index,order))
		ghost_frag_index=[]
		for i in range (0, len(real_frag_index)):
			ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))

		i =0	
		while(i< len(real_frag_index)):
	#	for i in range (0, len(real_frag_index)):
			pyscfatomstring=""
			mol = gto.Mole()
			for j in range (0, order):
				for k in range (0, self.atom_group):
					s = self.coords[real_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+str(self.AtomName(real_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";"	
			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					s = self.coords[ghost_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+"GHOST"+str(j*self.atom_group+k)+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";" 
			pyscfatomstring=pyscfatomstring[:-1]+"  "
			mol.atom =pyscfatomstring			
		
			mol.basis ={}
			ele_set = list(set(self.AllAtomNames()))
			for ele in ele_set:
				mol.basis[str(ele)]="cc-pvqz"

			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					atom_type = self.AtomName(ghost_frag_index[i][j]*self.atom_group+k)
					mol.basis['GHOST'+str(j*self.atom_group+k)]=gto.basis.load('cc-pvqz',str(atom_type))
			mol.verbose=0
			try:
				print "doing case ", i 
				time_log = time.time()
				mol.build()
				mf=scf.RHF(mol)
				hf_en = mf.kernel()
				mp2 = mp.MP2(mf)
				mp2_en = mp2.kernel()
				en = hf_en + mp2_en[0]
				#print "hf_en", hf_en, "mp2_en", mp2_en[0], " en", en	
				self.frag_mbe_energies[LtoS(real_frag_index[i])]=en
				print ("pyscf time..", time.time()-time_log)
				i = i+1
				gc.collect()
			except Exception as Ex:
				print "PYSCF Calculation error... :",Ex
				print "Mol.atom:", mol.atom
				print "Pyscf string:", pyscfatomstring
		return 0

	def PySCF_Frag_MBE_Energy_All(self):
		for i in range (0, self.FragOrder):
			self.PySCF_Frag_MBE_Energy(i+1)
		return  0


	def Set_Frag_MBE_Energy(self):
		self.frag_mbe_energy =  self.Frag_MBE_Energy()
		prod = 1
		for i in self.dist:
			prod = i*prod
		print "self.frag_mbe_energy", self.frag_mbe_energy
		return 0

	def Frag_MBE_Energy(self,  index=None):     # Get MBE energy recursively 
		if index==None:
			index=range(0, self.FragOrder)
		order = len(index)
		if order==0:
			return 0
		energy = self.frag_mbe_energies[LtoS(index)] 
		for i in range (0, order):
			sub_index = list(itertools.combinations(index, i))
			for j in range (0, len(sub_index)):
				energy=energy-self.Frag_MBE_Energy( sub_index[j])
		return  energy

	def CopyTo(self, target):
                target.FragOrder = self.FragOrder
                target.frag_mbe_energies=self.frag_mbe_energies
                target.frag_mbe_energy = self.frag_mbe_energy
                target.permute_index = self.permute_index

	def Permute_Frag_by_Index(self, index):
		new_frag = Frag( atoms_ =  self.atoms, coords_ = self.coords, index_= self.index, dist_=self.dist, atom_group_=self.atom_group)
		self.CopyTo(new_frag)
		new_frag.permute_index = index	
		
		new_frag.coords=new_frag.coords.reshape((new_frag.FragOrder, new_frag.atom_group,  -1))
		new_frag.coords = new_frag.coords[new_frag.permute_index]
		new_frag.coords = new_frag.coords.reshape((new_frag.FragOrder*new_frag.atom_group, -1))

		new_frag.atoms = new_frag.atoms.reshape((new_frag.FragOrder, new_frag.atom_group))  
                new_frag.atoms = new_frag.atoms[new_frag.permute_index]
                new_frag.atoms = new_frag.atoms.reshape(new_frag.FragOrder*new_frag.atom_group)

		# needs some code that fix the keys in frag_mbe_energies[LtoS(index)] after permutation in futher.  KY	

		return new_frag

	def Permute_Frag(self):
		permuted_frags=[]
		indexs=list(itertools.permutations(range(0, self.FragOrder)))
		for index in indexs:
			permuted_frags.append(self.Permute_Frag_by_Index(list(index)))
		return permuted_frags 

	def Frag_Force(self, cm_deri, nn_deri):
		return self.Combine_CM_NN_Deri(cm_deri, nn_deri)	

	def Combine_CM_NN_Deri(self, cm_deri, nn_deri):
		natom = self.NAtoms()
		frag_deri = np.zeros((natom, 3))
		for i in range (0, natom):
			for j in range (0, natom):
				if j >= i:
					cm_dx = cm_deri[i][j][0]
					cm_dy = cm_deri[i][j][1]
					cm_dz = cm_deri[i][j][2] 
					nn_deri_index = i*(natom+natom-i+1)/2 + (j-i)
					nn_dcm = nn_deri[nn_deri_index]
				else:
					cm_dx = cm_deri[j][i][3]
                                        cm_dy = cm_deri[j][i][4]
                                        cm_dz = cm_deri[j][i][5]
					nn_deri_index = j*(natom+natom-j+1)/2 + (i-j)
					nn_dcm = nn_deri[nn_deri_index]
				frag_deri[i][0] += nn_dcm * cm_dx
				frag_deri[i][1] += nn_dcm * cm_dy
				frag_deri[i][2] += nn_dcm * cm_dz
		return frag_deri
