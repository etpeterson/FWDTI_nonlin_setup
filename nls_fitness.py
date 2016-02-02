def _nls_fitness(S,W,g,l,b,Diso=3e-3):
    #setup some helper variables
    S0=S[0]  # number. correct?
    Si=S[1:]  # vector. correct?
    f=l[-1]  # number. correct?
    l=l[0:-1]  # vector. correct?
    Wg=np.dot(W,g)  # matrix*vector=vector. correct?
    ebDiso=np.exp(-b*Diso)  # number
    eWg=np.exp(Wg)  # vector

    #calculate the fitness
    F=(Si-(1-f)*eWg-f*S0*ebDiso)
    F=0.5*np.dot(F,F.T)

    #calculate the jacobian
    jac_base=Si-f*S0*ebDiso-(1-f)*eWg  # vector
    jac_lambdas=(f-1)*F*eWg*W  # this should be a vector times a matrix to result in a vector
    jac=np.array([0.5*(-2*S0*ebDiso+2*eWg)*jac_base,jac_lambdas*W[0,:],jac_lambdas*W[1,:],jac_lambdas*W[2,:],
                  jac_lambdas*W[3,:],jac_lambdas*W[4,:],jac_lambdas*W[5,:]])  # all this should result in a vector

    #calculate the hessian
    ebDisoWg=np.exp(-b*Diso+Wg)
    hess_lambdas=(f-1)**2*np.exp(b*Diso)*eWg*ebDisoWg+(f-1)*(-S0*f+((f-1)*eWg+Si)*ebDiso)*ebDisoWg  # should be a vector, just multiply by W[A,:]*W[B,:]
    hess_f=np.array([-(S0-ebDisoWg)*(f-1)*ebDiso*eWg,
                     -(S0*f-((f-1)*eWg+Si)*np.exp(b*Diso))*np.exp(-2*b*Diso)*np.exp(b*Diso+Wg)])  # two vectors. multiply by max(row,col)
    hess_elem1=(S0-np.exp(b*Diso)*eWg)*(S0-np.exp(b*Diso+Wg))*np.exp(-2*bDiso)
    hess_row1_col1=hess_f[0]*W[1:,:]+hess_f[2]*W[1:,:]
    hess_rest=hess_lambdas*W*W  # ???? see above for what this should be
    hess=np.concatenate((np.concatenate((hess_elem1,hess_row1_col1),axis=0),np.concatenate((hess_row1_col1,hess_rest),axis=0)),axis=1)

    #calculate the delta
    delta=-np.dot(np.linalg.inv(hess+l*np.eye(8)),jac)
    return F, jac, hess, delta