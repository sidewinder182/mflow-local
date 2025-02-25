
import copy
import time

class node():
    def __init__(self, function=None, args=[], kwargs={}, name=None, parents=[]):
        self.function = function
        self.args = list(args)
        self.kwargs = kwargs
        self.name = name
        self.parents=parents
        self.out = None
        self.args_parents = {}
        self.kwargs_parents = {}
        
        self.long_name = ""

        #Trace parents and store
        for i,arg in enumerate(self.args):
            if type(arg)==type(self):
                # print(arg)
                self.args_parents[i]  = arg
                self.long_name += arg.long_name + "." 
                
        for kw in self.kwargs:
            if type(self.kwargs[kw])==type(self):
                self.kwargs_parents[kw]=self.kwargs[kw]
                self.long_name += self.kwargs[kw].long_name + "." 

    def run(self):
        time1 = time.time()
        # if self.out is not None:
            #If function at this node has already been computed, 
            #just return the value
            # return(self.out)
        # else:
            #If function has not been computed, first compute all
            #arguments that are also nodes, then compute
            #the function itself and store the result.
            
        for i in self.args_parents:
            #self.args[i]=self.args_parents[i].run()
            self.args[i]=self.args_parents[i].out
        for kw in self.kwargs_parents:
            #self.kwargs[kw]=self.kwargs_parents[kw].run()
            self.kwargs[kw]=self.kwargs_parents[kw].out()
        self.out =  self.function(*self.args, **self.kwargs)
        print("Time taken by node " + str(self.name) + " : " + str(time.time()-time1))
        return (self.out)
            
    def get_args(self):
        for i in self.args_parents:
            #self.args[i]=copy.copy(self.args_parents[i].out)
            self.args[i]=self.args_parents[i].out
        return  self.args   
        
    def get_kwargs(self): 
        for kw in self.kwargs_parents:
            #self.kwargs[kw]=copy.copy(self.kwargs_parents[kw].out)
            self.kwargs[kw]=self.kwargs_parents[kw].out      
        return  self.kwargs


class pipelineNode():
    def __init__(self, initGraph, node_list, id):
        self.id = id
        self.head = node_list[0]
        self.tail = node_list[len(node_list)-1]
        self.name = ""
        self.node_list = node_list
        self.initGraph = initGraph
        # function_list = []
        # print(node_list)
        for i, id in enumerate(node_list):
            self.name += initGraph.node[id]["block"].name
            # print([initGraph.node[id]["block"].function])
            # function_list.append(initGraph.node[id]["block"].function)
            if i != len(node_list)-1:
                self.name += "."
        # function_list.reverse()
        # print(function_list)
        # self.function = lambda x : function_list[0](x)
        # for func in function_list[1:]:
        #     self.function = lambda x : func(self.function(x))
        self.out = None

    def run(self):
        time1 = time.time()
        if self.out is not None:
            #If function at this node has already been computed, 
            #just return the value
            return(self.out)
        else:
            #If function has not been computed, first compute all
            #arguments that are also nodes, then compute
            #the function itself and store the result.
            
            # for i in self.args_parents:
            #     #self.args[i]=self.args_parents[i].run()
            #     self.args[i]=self.args_parents[i].out
            # for kw in self.kwargs_parents:
            #     #self.kwargs[kw]=self.kwargs_parents[kw].run()
            #     self.kwargs[kw]=self.kwargs_parents[kw].out()
            # self.out =  self.function(*self.args, **self.kwargs)
            for node_id in self.node_list:
                print(node_id)
                intermediate_out = self.initGraph.node[node_id]["block"].run()
            self.out = intermediate_out
            print("Time taken by "+ str(self.initGraph.node[node_id]["block"].name) + " : " + str(time.time()-time1))
            return (self.out)

            