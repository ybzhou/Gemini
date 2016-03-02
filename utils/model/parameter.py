import copy

class Parameter:
    def __init__(self, params):
        """params is a dictionary with key being the name of the parameter and 
           value being the actual parameter, default all parameters to be tunable
           but without any regularization and constraint"""
           
        self.params = {}
        self.lr_scheduler = None
        self.mu_scheduler = None
        for pname in params.keys():
            self.params[pname] = {'value': params[pname],
                                  'tune': True,
                                  'regularizer': None,
                                  'constraint': None,
                                  'learning_rate': None,
                                  'momentum': None}
    
    def addParameters(self, params, tune,
                     regularizer, constraint,
                     learning_rate, momentum):
        for pname in params.keys():
            self.params[pname] = {'value': params[pname],
                                  'tune': tune[pname],
                                  'regularizer': regularizer[pname],
                                  'constraint': constraint[pname],
                                  'learning_rate': learning_rate[pname],
                                  'momentum': momentum[pname]}
    
    def removeParameter(self, pname):
        self.params.pop(pname)
    
    def setLearningRate(self, pname, lr):
        self.params[pname]['learning_rate'] = lr
        
    def getLearningRate(self, pname):
        return self.params[pname]['learning_rate']
    
    def setLearningRateScheduler(self, scheduler):
        self.lr_scheduler = scheduler
    
    def getLearningRateScheduler(self):
        return self.lr_scheduler
    
    def setMomentum(self, pname, mu):
        self.params[pname]['momentum'] = mu
        
    def getMomentum(self, pname):
        return self.params[pname]['momentum']
    
    def setMomentumScheduler(self, scheduler):
        self.mu_scheduler = scheduler
    
    def getMomentumScheduler(self):
        return self.mu_scheduler
    
    def getAllParameterNames(self):
        return self.params.keys()
    
    def getTunableParameternames(self):
        name = []
        for p in self.params.keys():
            para = self.params[p]
            if para['tune']:
                name.append(p)
        return name
        
    def getParameter(self, pname):
        return self.params[pname]['value']
    
    def getParameterValue(self, pname):
        return self.params[pname]['value'].get_value()
    
    def setParameter(self, pname, paramExpr):
        self.params[pname]['value'] = paramExpr
    
    def setParameterValue(self, pname, paramValue):
        self.params[pname]['value'].set_value(paramValue, borrow=True)
    
    def getRawParameters(self):
        return self.params
    
    def setRawPrarmeters(self, params):
        self.params = copy.deepcopy(params)
    
    def getAllParameters(self):
        param = [p['value'] for p in self.params]
        return param
    
    def getNumTunableParams(self):
        cnt = 0
        for p in self.params.keys():
            para = self.params[p]
            if para['tune']:
                cnt += 1
        return cnt
    
    def getTotalNumParams(self):
        return len(self.params.keys())
    
    def setTunableParameters(self, tunable):
        for k in tunable.keys():
            self.params[k]['tune'] = tunable[k]
            
    def getTunableParameters(self):
        param = []
        for pn in self.params.keys():
            if self.params[pn]['tune']:
                param.append(self.params[pn]['value'])
        return param
    
    def setParamRegularization(self, reg):
        for k in reg.keys():
            self.params[k]['regularizer'] = reg[k]
            
    def getParamRegularization(self):
        reg = 0.
        for pn in self.params.keys():
            if self.params[pn]['regularizer'] is not None:
                reg += self.params[pn]['regularizer'].getRegularization(self.params[pn]['value'])
        return reg
    
    def setParamRegularizationValue(self, pname, val):
        self.params[pname]['regularizer'].setRegularizationCoef(val)
        
    def getParamRegularizationValue(self, pname):
        return self.params[pname]['regularizer'].getRegularizationCoef()
    
    def getRegularizedParamNames(self):
        return [pname for pname in self.params if self.params[pname]['regularizer'] is not None]
    
    def setParamConstraint(self, constraint):
        for k in constraint.keys():
            self.params[k]['constraint'] = constraint[k]
            
    def applyParamConstraint(self, pname, value):
        p = self.params[pname]
        updated_value = value
        if p['constraint'] is not None:
            updated_value = p['constraint'].applyConstraint(value)
        return updated_value
    
    