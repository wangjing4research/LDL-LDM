# LDL-LDM
Code for our TNNLS'21 paper titled Label Distribution Learning by Exploiting Label Distribution Manifold

Our paper presents a new LDL method which exploits both global and local label correlations. It uses the manifold strucutre of label distribution to model label correaltion, which doesn't rely on any assumption and is totally data-dependent. 

![Framework of LDL-LDM](./framework.png)

Use our code and cite
>@article{wang_label_2023, \
	title = {Label distribution learning by exploiting label distribution manifold}, \
	journal = {IEEE Transactions on Neural Networks and Learning Systems},\
	author = {Wang, Jing and Geng, Xin},\
	year = {2023},\
	volume={34}, \
  	number={2}, \
	pages={839-852}, \
>}



# How to use
LDL-LDM with full LDL: python ldm_full.py

LDL-LDM with missing LDL: python ldm_incom.py

# Comparing methods

IIS-LLD and SA-BFGS: http://ldl.herokuapp.com/LDLPackage_v1.2.zip

Adam-LDL-SCL, EDL-LRL, and LDL-LCLR, refer to https://github.com/NJUST-IDAM/

IncomLDL: http://www.lamda.nju.edu.cn/files/IncomLDL.zip

