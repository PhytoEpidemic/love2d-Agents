
local NN = {}
local agent = ...
local table = table
table.add = function(tab1,tab2)
	for i=#tab1+1,#tab1+#tab2 do
		tab1[i] = tab2[i-#tab1]
	end
	return tab1
end
local function updateStatus(params)
	love.thread.getChannel( agent.id.."status" ):push(params)
	love.thread.getChannel( agent.id.."status" ):pop()
end
local Activations = {
	
	ReLU = {
		func = function(x)
			return math.max(0,x)
		end,
		derivative = function(x)
			if x >= 0 then return 1 else return 0 end
		end,
	},
	GeLU = {
		func = function(x)
			return x * 0.5 * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
		end,
		derivative = function(x)
			local a = 0.5 * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
			local b = 0.5 * x * ((1 / math.cosh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))^2) * (0.0356774 * math.pow(x, 2) + math.sqrt(2 / math.pi))
			return a + b
		end,
	},

	Tanh = {
		func = function(x)
			return math.tanh(x)
		end,
		derivative = function(x)
			return 1/math.cosh(x)
		end,
	},
	lReLU = {
		func = function(x)
			if x >= 0 then
				return x
			else
				return x*.01
			end
		end,
		derivative = function(x)
			if x >= 0 then return 1 else return .01 end
		end,
	},
	LeakyReLU = {
		alpha = 0.01, -- You can choose a different value for alpha if desired
		func = function(x)
			return x >= 0 and x or Activations.LeakyReLU.alpha * x
		end,
		derivative = function(x)
			return x >= 0 and 1 or Activations.LeakyReLU.alpha
		end,
	},
	
	slReLU = {
		func = function(x)
			if x >= 0 then
				return x
			else
				return x*.1
			end
		end,
		derivative = function(x)
			if x >= 0 then return 1 else return .1 end
		end,
	},
	
	sigmoid = {
		func = function(x)
			return (1/(1+math.exp(-x)))
		end,
		derivative = function(x) return x*(1-x) end
	},
	tanhh = {
		func = function(x)
			return math.tanh(x)
		end,
		derivative = function(x)
			return 1/math.cosh(x)
		end,
	},
	logflow = function(x)
		local Flip = false
		if x <= 0 then
			x = -x
			Flip = true
		end
		x = math.log(x)+x^-x
		if Flip then
			return -x
		else
			return x
		end
		
	end,
	logthink = function(x)
		local Flip = false
		if x <= 0 then
			x = -x
			Flip = true
		end
		x = math.log(x)
		if Flip then
			return -x
		else
			return x
		end
		
	end,
	house = function(x,con)
		x = x%con*2
		if x > con then	
			x = x-(x-con)*2
		end
		return x/con
	end,
	housed = function(x,con)
		x = x%con*2
		if x > con then	
			return -1
		else
			return 1
		end
	end,
	

	
}


local Optimizers = {
	sgd = function(tab)
		return tab[1]
	end,
	batch = function(tab)
		return NN:getAverage(tab)
	end,
}

local Architecture = {}
	Architecture.Blank = {
		predict = function(layer,data)
			
		end,
		forwardPass = function(layer,data)
			
		end,
		new = function(layer)
			
		end,
		getErrors = function(layer1,layer2,target)
			
		end,
		getDeltas = function(layer)
			
		end,
		addDeltas = function(layer)
			
		end,
		clearDeltas = function(layer)
			
		end,
		clean = function(layer)
			
		end,
	}
	Architecture.FF = {
		predict = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				for spot,weight in ipairs(neuron.W) do
					total = total+data[spot]*weight
				end
				newData[n] = Act.func(total)
			end
			return newData
		end,
		forwardPass = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				for spot,weight in ipairs(neuron.W) do
					total = total+data[spot]*weight
				end
				total = Act.func(total)
				newData[n] = total
				neuron.Output = total
			end
			return newData
		end,
		
		new = function(layer)
			if not layer.wvar then
				layer.wvar = .5
			end
			if not layer.bvar then
				layer.bvar = .5
			end
			local parameterCount = 0
			local init = math.sqrt(1/layer.neurons)
			local generateWeights = function(num)
				local W = {}
				parameterCount = parameterCount+num
				for _=1,num do
					local initalWeight = init*NN:gaussian(0,layer.wvar)
					if layer.Act == "ReLU" or layer.Act == "GeLU" then
						initalWeight = math.abs(initalWeight)
					end
					table.insert(W, initalWeight)
				end
				return W
			end
			local neuronLayer = {}
			parameterCount = parameterCount+layer.neurons
			for _=1,layer.neurons do
				neuron = {}
				neuron.W = generateWeights(layer.shape)
				neuron.B = math.abs(NN:gaussian(0,layer.bvar)*init)
				table.insert(neuronLayer,neuron)
			end
			return neuronLayer, parameterCount
		end,
		getErrors = function(layer1,layer2,target)
			local err
			local function getHiddenErrors(n)
				for _,n2 in ipairs(layer2.Net) do
					err = err+(n2.Error*n2.W[n])
				end
			end
			for n,neuron in ipairs(layer1.Net) do
				err = 0
				if target then
					--print(target[n],neuron.Output)
					err = target[n]-neuron.Output
				else
					getHiddenErrors(n)
				end
				neuron.Error = err
			end
		end,
		getDeltas = function(layer)
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local gradi = Act.derivative(neuron.Output)*neuron.Error
				table.insert(neuron.BiasDeltas,gradi)
				for spot,_ in ipairs(neuron.W) do
					table.insert(neuron.WeightDeltas[spot], layer.Inputs[spot]*gradi)
				end
			end
		end,
		addDeltas = function(layer)
			local Opt = Optimizers[layer.Opt]
			for n,neuron in ipairs(layer.Net) do
				for i,tab in ipairs(neuron.WeightDeltas) do
					local delta = Opt(tab)*layer.Lr
					if tostring(delta) ~= "nan" then
						neuron.W[i] = neuron.W[i] + delta
					end
				end
				local delta = Opt(neuron.BiasDeltas)*layer.Lr
				if tostring(delta) ~= "nan" then
					neuron.B = neuron.B+delta
				end
			end
		end,
		clearDeltas = function(layer)
			for _,neuron in ipairs(layer.Net) do
				neuron.WeightDeltas = {}
				for w,_ in ipairs(neuron.W) do
					neuron.WeightDeltas[w] = {}
				end
				neuron.BiasDeltas = {}
			end
		end,
		clean = function(layer)
			for n,neuron in ipairs(layer.Net) do
				local temp = NN:deepCopy(layer.Net[n])
				layer.Net[n] = {}
				layer.Net[n]["W"] = temp.W
				layer.Net[n]["B"] = temp.B
			end
		end,
	}
	Architecture.RN = {
		predict = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				for spot=1,layer.shape do
					total = total+data[spot]*neuron.W[spot]
				end
				for n2=layer.shape+1,layer.shape+layer.neurons do
					total = total+layer.Net[n2-layer.shape]["S"]*neuron.W[n2]
				end
				newData[n] = Act.func(total)
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = newData[n]
			end
			return newData
		end,
		forwardPass = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				for spot=1,layer.shape do
					total = total+data[spot]*neuron.W[spot]
				end
				for n2=layer.shape+1,layer.shape+layer.neurons do
					total = total+layer.Net[n2-layer.shape]["S"]*neuron.W[n2]
					layer.Inputs[n2] = layer.Net[n2-layer.shape]["S"]
				end
				total = Act.func(total)
				newData[n] = total
				neuron.Output = total
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = newData[n]
			end
			return newData
		end,
		
		new = function(layer)
			if not layer.wvar then
				layer.wvar = .5
			end
			if not layer.bvar then
				layer.bvar = .5
			end
			local parameterCount = 0
			local init = math.sqrt(1/layer.neurons)
			local generateWeights = function()
				local W = {}
				parameterCount = parameterCount+(layer.shape+layer.neurons)
				for _=1,layer.shape+layer.neurons do
					table.insert(W, init*NN:gaussian(0,layer.wvar))
				end
				return W
			end
			local neuronLayer = {}
			parameterCount = parameterCount+layer.neurons
			for _=1,layer.neurons do
				neuron = {}
				neuron.W = generateWeights()
				neuron.B = NN:gaussian(0,layer.bvar)*init
				neuron.S = 0
				table.insert(neuronLayer,neuron)
			end
			return neuronLayer, parameterCount
		end,
		getErrors = Architecture.FF.getErrors,
		getDeltas = function(layer)
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local gradi = Act.derivative(neuron.Output)*neuron.Error
				table.insert(neuron.BiasDeltas,gradi)
				for spot,_ in ipairs(neuron.W) do
					table.insert(neuron.WeightDeltas[spot], layer.Inputs[spot]*gradi)
				end
			end
		end,
		addDeltas = function(layer)
			local Opt = Optimizers[layer.Opt]
			for n,neuron in ipairs(layer.Net) do
				for i,tab in ipairs(neuron.WeightDeltas) do
					local delta = Opt(tab)*layer.Lr
					if tostring(delta) ~= "nan" then
						neuron.W[i] = neuron.W[i] + delta
					end
				end
				local delta = Opt(neuron.BiasDeltas)*layer.Lr
				if tostring(delta) ~= "nan" then
					neuron.B = neuron.B+delta
				end
			end
		end,
		clearDeltas = function(layer)
			for _,neuron in ipairs(layer.Net) do
				neuron.WeightDeltas = {}
				for w,_ in ipairs(neuron.W) do
					neuron.WeightDeltas[w] = {}
				end
				neuron.BiasDeltas = {}
				neuron["S"] = 0
			end
		end,
		clean = function(layer)
			for n,neuron in ipairs(layer.Net) do
				local temp = NN:deepCopy(layer.Net[n])
				layer.Net[n] = {}
				layer.Net[n]["W"] = temp.W
				layer.Net[n]["B"] = temp.B
				layer.Net[n]["S"] = 0
			end
		end,
		forget = function(layer)
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = 0
			end
		end,
	}
	Architecture.LRN = {
		predict = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				for spot=1,layer.shape do
					total = total+data[spot]*neuron.W[spot]
				end
				local itor = 0
				for n2=layer.shape+1,layer.shape+layer.neurons do
					itor = itor+1
					total = total+layer.Net[itor]["S"]*neuron.W[n2]
				end
				itor = 0
				for n3=layer.shape+layer.neurons+1,layer.shape*2+layer.neurons do
					itor = itor+1
					total = total+layer.Net[itor]["LS"]*neuron.W[n3]
				end
				newData[n] = Act.func(total)
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = newData[n]
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["LS"] = neuron["LS"]+newData[n]
			end
			return newData
		end,
		forwardPass = function(layer,data)
			local newData = layer.Output
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local total = neuron.B
				
				for spot=1,layer.shape do
					total = total+data[spot]*neuron.W[spot]
				end
				local itor = 0
				for n2=layer.shape+1,layer.shape+layer.neurons do
					itor = itor+1
					total = total+layer.Net[itor]["S"]*neuron.W[n2]
					layer.Inputs[n2] = layer.Net[itor]["S"]
				end
				itor = 0
				for n3=layer.shape+layer.neurons+1,layer.shape*2+layer.neurons do
					itor = itor+1
					total = total+layer.Net[itor]["LS"]*neuron.W[n3]
					layer.Inputs[n3] = layer.Net[itor]["LS"]
				end
				total = Act.func(total)
				newData[n] = total
				neuron.Output = total
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = newData[n]
			end
			for n,neuron in ipairs(layer.Net) do
				neuron["LS"] = neuron["LS"]+newData[n]
			end
			return newData
		end,
		
		new = function(layer)
			if not layer.wvar then
				layer.wvar = .5
			end
			if not layer.bvar then
				layer.bvar = .5
			end
			local init = math.sqrt(1/layer.neurons)
			local generateWeights = function()
				local W = {}
				for _=1,layer.shape*2+layer.neurons do
					table.insert(W, init*NN:gaussian(0,layer.wvar))
				end
				return W
			end
			local neuronLayer = {}
			for _=1,layer.neurons do
				neuron = {}
				neuron.W = generateWeights()
				neuron.B = NN:gaussian(0,layer.bvar)*init
				neuron.S = 0
				neuron.LS = 0
				table.insert(neuronLayer,neuron)
			end
			return neuronLayer
		end,
		getErrors = Architecture.FF.getErrors,
		getDeltas = function(layer)
			local Act = Activations[layer.Act]
			for n,neuron in ipairs(layer.Net) do
				local gradi = Act.derivative(neuron.Output)*neuron.Error
				table.insert(neuron.BiasDeltas,gradi)
				for spot,_ in ipairs(neuron.W) do
					table.insert(neuron.WeightDeltas[spot], layer.Inputs[spot]*gradi)
				end
			end
		end,
		addDeltas = function(layer)
			local Opt = Optimizers[layer.Opt]
			for n,neuron in ipairs(layer.Net) do
				for i,tab in ipairs(neuron.WeightDeltas) do
					local delta = Opt(tab)*layer.Lr
					if tostring(delta) ~= "nan" then
						neuron.W[i] = neuron.W[i] + delta
					end
				end
				local delta = Opt(neuron.BiasDeltas)*layer.Lr
				if tostring(delta) ~= "nan" then
					neuron.B = neuron.B+delta
				end
			end
		end,
		clearDeltas = function(layer)
			for _,neuron in ipairs(layer.Net) do
				neuron.WeightDeltas = {}
				for w,_ in ipairs(neuron.W) do
					neuron.WeightDeltas[w] = {}
				end
				neuron.BiasDeltas = {}
				neuron["S"] = 0
				neuron["LS"] = 0
			end
		end,
		clean = function(layer)
			for n,neuron in ipairs(layer.Net) do
				local temp = NN:deepCopy(layer.Net[n])
				layer.Net[n] = {}
				layer.Net[n]["W"] = temp.W
				layer.Net[n]["B"] = temp.B
				layer.Net[n]["S"] = 0
				layer.Net[n]["LS"] = 0
			end
		end,
		forget = function(layer)
			for n,neuron in ipairs(layer.Net) do
				neuron["S"] = 0
				neuron["LS"] = 0
			end
		end,
	}





	function NN:new(layers,params)
		if type(layers) ~= "table" then
			error("First Arg must be table. got: "..type(layers))
		end
		local model = self:deepCopy(layers)
		local parameterCount = 0
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			if type(params) == "table" then
				for key,arg in pairs(params) do
					if not layer[key] then
						layer[key] = arg
					end
				end
			end
			if l == 1 then
				if not layer.shape then
					error("shape must be given to first layer.")
				end
			end
			if not layer.shape then
				layer.shape = model[l-1]["neurons"]
			end
			layer.Output = {}
			layer.Net, layer.parameterCount = Arc.new(layer)
			parameterCount = layer.parameterCount
		end
		model.parameterCount = parameterCount
		return model
	end
	function NN:updateParams(model,params,Layer)
		local up = function(layer)
			if params then
				for key,arg in pairs(params) do
					layer[key] = arg
				end
			end
		end
		for l,layer in ipairs(model) do
			if Layer then	
				if Layer == l then
					up(layer)
				end
			else
				up(layer)
			end
		end
	end
	function NN:predict(model,data)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			data = Arc.predict(layer,data)
		end
		model.Output = data
		return model.Output
	end
	function NN:forwardPass(model,data)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			layer.Inputs = self:deepCopy(data)
			data = Arc.forwardPass(layer,data)
		end
		model.Output = data
		return model.Output
	end
	function NN:getErrors(model,target)
		for l=#model,1,-1 do
			local Arc = Architecture[model[l]["Arc"]]
			local layer2 = false
			if model[l+1] then
				layer2 = model[l+1]
			end
			local tar = false
			if l == #model then
				tar = target
			end
			Arc.getErrors(model[l],layer2,tar)
		end
	end
	function NN:getDeltas(model)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			Arc.getDeltas(layer)
		end
	end
	function NN:backPropagate(model,target)
		self:getErrors(model,target)
		self:getDeltas(model)
	end
	function NN:addDeltas(model)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			Arc.addDeltas(layer)
		end
	end
	function NN:clearDeltas(model)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			Arc.clearDeltas(layer)
		end
	end
	function NN:forget(model)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			pcall(Arc.forget, layer)
		end
	end
	function NN:clean(model)
		for l,layer in ipairs(model) do
			local Arc = Architecture[layer.Arc]
			Arc.clean(layer)
			layer.Output = {}
			layer.Inputs = {}
		end
	end
	function NN:deepCopy(original)
		if type(original) ~= "table" then
			return error("Must be a table to copy.   got: "..type(original))
		end
		local copy = {}
		for key, value in pairs(original) do
			if type(value) == "table" then
				copy[key] = self:deepCopy(value)
			else
				copy[key] = value
			end
		end
		return copy
	end
	function NN:saveTable(file,tab)
		local f = io.output(file)
		for _,data in ipairs(tab) do
			f.write(f,data,"\n")
		end
		io.close(f)
	end
	function NN:gaussian(mean, variance)
		return math.sqrt(-2 * variance * math.log(math.random())) * math.cos(2 * math.pi * math.random()) + mean
	end
	
	function NN:tostring(t, name, indent)
		local cart
		local autoref
		local bytes = 0
		local function isemptytable(t) return next(t) == nil end
		local function basicSerialize (o)
			local so = tostring(o)
			if type(o) == "function" then
				local info = debug.getinfo(o, "S")
				-- info.name is nil because o is not a calling level
				if info.what == "C" then
					return string.format("%q", so .. ", C function")
				else
					-- the information is defined through lines
					return string.format("%q", so .. ", defined in (" ..
					info.linedefined .. "-" .. info.lastlinedefined ..
					")" .. info.source)
				end
			elseif type(o) == "number" or type(o) == "boolean" then
				return so
			else
				return string.format("%q", so)
			end
		end
		local function insertData(data)
			bytes = bytes+#data
			table.insert(cart,data)
		end
		local function addtocart (value, name, indent, saved, field)
			indent = indent or ""
			saved = saved or {}
			field = field or name
			local item = indent .. field
			if type(value) ~= "table" then
				item = item .. " = " .. basicSerialize(value) .. ";"
				insertData(item)
			else
				if saved[value] then
					item = item .. " = {}; -- " .. saved[value]
					.. " (self reference)\n"
					local ref = name .. " = " .. saved[value] .. ";"
					bytes = bytes+#ref
					table.insert(autoref,ref)
					insertData(item)
				else
					saved[value] = name
					if isemptytable(value) then
						item = item .. " = {};"
						insertData(item)
					else
						item = item .. " = {"
						insertData(item)
						item = ""
						for k, v in pairs(value) do
							k = basicSerialize(k)
							local fname = string.format("%s[%s]", name, k)
							field = string.format("[%s]", k)
							-- three spaces between levels
							addtocart(v, fname, indent .. "\t", saved, field)
						end
						item = item .. indent .. "};"
						insertData(item)
					end
				end
			end
			updateStatus({status = "Saving",bytes = bytes})
		end
		name = name or "__unnamed__"
		if type(t) ~= "table" then
			return name .. " = " .. basicSerialize(t)
		end
		cart, autoref = {}, {}
		addtocart(t, name, indent)
		return table.concat(cart, "\n") .. table.concat(autoref, "\n")
	end

	function NN:softMax(tab)
		local t = 0
		for _,value in ipairs(tab) do
			t = t+value
		end
		local newTab = {}
		for _,value in ipairs(tab) do
			if t == 0 then
				table.insert(newTab,0)
			else
				table.insert(newTab,value/t)
			end
		end
		return newTab
	end
	function NN:initDataSet()
		return {}
	end
	function NN:addData(set,d,t)
		dta = {}
		dta.Data = d
		dta.Target = t
		table.insert(set,dta)
	end
	function NN:getAverage(tab)
		local t = 0
		for _,value in ipairs(tab) do
			t = t+(value or 0)
		end
		if tostring(t) == "nan" then
			return 0
		else
			return t/math.max(1,#tab)
		end
		
	end
	function NN:getTotal(tab)
		local t = 0
		for _,value in ipairs(tab) do
			t = t+value
		end
		return t
	end
	function NN:calculateAverageFitness(model,data)
		local dataS = {}
		local dataT = {}
		for i,s in ipairs(data) do
			table.insert(dataS,s.Data)
			table.insert(dataT,s.Target)
		end
		local allfits = {}
		for i=1,#dataS do
			re = self:predict(model,dataS[i])
			cdataT = dataT[i]
			fits = {}
			for r,result in ipairs(re) do
				fits[r] = math.abs(cdataT[r]-result)--self:calculateLoss(cdataT[r],result)
			end
			allfits[i] = self:getAverage(fits)
		end
		return self:getAverage(allfits)--(1-self:getAverage(allfits)*2)*100
	end
	function NN:calculateFullFitness(model,data)
		local dataS = {}
		local dataT = {}
		for i,s in ipairs(data) do
			table.insert(dataS,s.Data)
			table.insert(dataT,s.Target)
		end
		local allfits = 0
		for i=1,#dataS do
			re = self:predict(model,dataS[i])
			cdataT = dataT[i]
			fits = {}
			for r,result in ipairs(re) do
				fits[r] = math.abs(cdataT[r]-result)--self:calculateLoss(cdataT[r],result)
			end
			allfits = allfits+self:getTotal(fits)
		end
		return allfits--(1-self:getAverage(allfits)*2)*100
	end
	function NN:calculateLossFitness(re,dataT)
			local fits = {}
			for r,result in ipairs(re) do
				fits[r] = self:calculateLoss(dataT[r],result)
			end
		return (1-self:getAverage(fits)*2)*100
	end
	function NN:calculateFitness(re,dataT)
			local fits = {}
			for r,result in ipairs(re) do
				fits[r] = math.abs(dataT[r]-result)
			end
		return (1-self:getAverage(fits))*100
	end
	function NN:calculateRealFitness(re,dataT)
			local fits = {}
			for r,result in ipairs(re) do
				if dataT[r] == 1 then
					if result > .5 then
						fits[r] = 1
					else
						fits[r] = 0
					end
				else
					if result < .5 then
						fits[r] = 1
					else
						fits[r] = 0
					end
				end
			end
		return (self:getAverage(fits))*100
	end
	function NN:calculateLoss(target,output)
		return ((target-output)^2)/2
	end
	
	function NN:netToString(model)
		return "local "..self:tostring(model,"model").."return model"
	end


NN.__index = NN



if type(agent) == "table" then
	
	math.randomseed(agent.Seed)
	if agent.Layers then
		agent.Model = NN:new(agent.Layers,agent.Model)
	end
	NN:clearDeltas(agent.Model)
	filesystem = require("love.filesystem")
	timer = require("love.timer")
	local Running = true
	local AgOptimizers = {
		batch = function(agent)
			NN:forget(agent.Model)
			local fits = {}
			if agent["Punishments"] then
				for d=#agent.Punishments,1,-1 do
					local pun = agent.Punishments[d]
					for IT=1,#pun do
						local p = pun[IT]
						--print("bad")
						local t = NN:deepCopy(p.T)
						for i,item in ipairs(t) do
							t[i] = 1-item
						end
						NN:forwardPass(agent.Model,p.I)
						NN:backPropagate(agent.Model,t)
						table.insert(fits,NN:calculateFitness(agent.Model.Output,t))
					end
					NN:forget(agent.Model)
				end
			end
			if agent["Rewards"] then
				for d=#agent.Rewards,1,-1 do
					local reward = agent.Rewards[d]
					for IT=1,#reward do
						local r = reward[IT]
						--print("good")
						NN:forwardPass(agent.Model,r.I)
						NN:backPropagate(agent.Model,r.T)
						table.insert(fits,NN:calculateFitness(agent.Model.Output,r.T))
					end
					NN:forget(agent.Model)
				end
			end
			NN:addDeltas(agent.Model)
			NN:clearDeltas(agent.Model)
			return NN:getAverage(fits)
		end,
		sgd = function(agent)
			NN:forget(agent.Model)
			local fits = {}
			if agent["Punishments"] then
				for d=#agent.Punishments,1,-1 do
					local pun = agent.Punishments[d]
					for IT=1,#pun do
						local p = pun[IT]
						--print("bad")
						local t = NN:deepCopy(p.T)
						for i,item in ipairs(t) do
							t[i] = 1-item
						end
						NN:forwardPass(agent.Model,p.I)
						NN:backPropagate(agent.Model,t)
						NN:addDeltas(agent.Model)
						NN:clearDeltas(agent.Model)
						table.insert(fits,NN:calculateFitness(agent.Model.Output,t))
					end
					
				end
				
			end
			if agent["Rewards"] then
				for d=#agent.Rewards,1,-1 do
					local reward = agent.Rewards[d]
					for IT=1,#reward do
						local r = reward[IT]
						--print("good")
						NN:forwardPass(agent.Model,r.I)
						NN:backPropagate(agent.Model,r.T)
						NN:addDeltas(agent.Model)
						NN:clearDeltas(agent.Model)
						table.insert(fits,NN:calculateFitness(agent.Model.Output,r.T))
					end
					
				end
				
			end
			return NN:getAverage(fits)
		end,
	}
	local RandomOptions = {
		confidence = function(agent)
			for i,item in ipairs(agent.Action) do
				local chance = math.random()*agent.Threshold
				local confidence = math.abs(item-.5)*2
				if confidence < chance then
					agent["Action"][i] = math.random(0,1)
			
				else
					--agent["Action"][i] = math.ceil(item-.5)
					--break
				end
			end
		end,
		greedy = function(agent)
			for i,item in ipairs(agent.Action) do
				local chance = math.random()
				if agent.Threshold > chance then
					agent["Action"][i] = math.random(0,1)
				end
			end
		end,
	}
	
	local CrossoverOptions = {
		two_point = function(agent,selection)
			if math.random() <= agent.CrossoverRate then
				for s,a in ipairs(selection) do	
					for l,layer in ipairs(agent.Model) do
						for n,neuron in ipairs(layer.Net) do
							for w,weight in ipairs(neuron.W) do
								if math.random() <= agent.CrossoverIntensity then
									a.Model[l]["Net"][n]["W"][w], weight = w, a.Model[l]["Net"][n]["W"][w]
									
								end
							end
							if math.random() <= agent.CrossoverIntensity then
								a.Model[l]["Net"][n]["B"], neuron.B = neuron.B, a.Model[l]["Net"][n]["B"]
							end
						end
					end
					break
				end
			end
		end,
	}
	
	Commands = {
		reward = function(self, args)
			if not agent["Rewards"] then
				agent["Rewards"] = {}
			end
			table.insert(agent["Rewards"],NN:deepCopy(agent.Past))
			agent.RPoints = agent.RPoints+args
		end,
		punish = function(self, args)
			if not agent["Punishments"] then
				agent["Punishments"] = {}
			end
			table.insert(agent["Punishments"],NN:deepCopy(agent.Past))
			agent.PPoints = agent.PPoints+args
		end,
		train = function(self, args, epochs)
			
			
			local recur = false
			if not epochs then
				epochs = epochs or 1
				local checkingqueue = epochs
				while checkingqueue do
					
					for q=1,love.thread.getChannel( agent["id"].."sent" ):getCount() do	
						local command = love.thread.getChannel( agent["id"].."sent" ):pop()
						if command[1] == "train" then
							epochs = epochs+1
						else
							love.thread.getChannel( agent["id"].."sent" ):push(command)
						end
					end
					if checkingqueue == epochs then
						checkingqueue = false
					else
						checkingqueue = epochs
						love.timer.sleep(0.1)
					end
				end
					
				updateStatus({status = "Training",completed = 0, epochs = epochs,Fitness = agent.Fitness})
				for e=epochs,1,-1 do
					self.train(self, args, e)
				end
				return
			end
			local fits = {}
			if agent.DataPool then
				local files = love.filesystem.getDirectoryItems( args.."/"..agent.DataPool )
				local totalD = #files
				
				for c=totalD,1,-1 do
					updateStatus({status = "Training",completed = math.floor((1-c/totalD)*100), epochs = epochs,Fitness = agent.Fitness})
					local rand = math.random(1,#files)
					local ch = love.filesystem.load(args.."/"..agent.DataPool.."/"..files[rand])
					if ch then
						local d = ch()
						for s,seq in ipairs(d.Rewards) do
							for r,rew in ipairs(seq) do
								for t,tar in pairs(rew.T) do
									d.Rewards[s][r]["T"][t] = math.max(math.min(tar,agent.TargetUpperLimit),agent.TargetLowerLimit)
								end
							end
						end
						agent["Punishments"] = d.Punishments
						agent["Rewards"] = d.Rewards
						NN:clearDeltas(agent.Model)
						local opt = AgOptimizers[agent.Opt]
						local fit = opt(agent)
						if fit > 0 then
							table.insert(fits,fit)
						else
							love.filesystem.remove(args.."/"..agent.DataPool.."/"..files[rand])
						end
						agent["Punishments"] = {}
						agent["Rewards"] = {}
						collectgarbage("collect")
						table.remove(files, rand)
					end
				end
			else
				NN:clearDeltas(agent.Model)
				local opt = AgOptimizers[agent.Opt]
				table.insert(fits,opt(agent))
				collectgarbage("collect")
			end
			agent.Fitness = NN:getAverage(fits)
		end,
		removeData = function(self, args)
			agent["Rewards"] = {}
			agent["Punishments"] = {}
		end,
		resetPoints = function(self, args)
			agent["PPoints"] = 0
			agent["RPoints"] = 0
		end,
		save = function(self, args)
			NN:clean(agent.Model)
			local file = love.filesystem.newFile(args)
			if file then
				file:open("w")
				file:write(NN:netToString(agent))
				file:close()
			end
			collectgarbage("collect")
		end,
		load = function(self, args)
			NN:clean(agent.Model)
			local file = love.filesystem.load(args)
			agent = file()
			collectgarbage("collect")
		end,
		forget = function(self, args)
			if agent.Record then
				agent.Past = {}
			end
			agent.Time = 0
			NN:forget(agent.Model)
			collectgarbage("collect")
		end,
		decay = function(self, params)
			for key,item in pairs(params) do
				if type(agent[key]) == "number" then
					local change = agent["Decay"..key]
					if type(params[item]) == "number" then
						change = params[item]
					end
					--print(agent[key])
					agent[key] = math.max(0,agent[key]-change)
					print(agent.id.." "..key..": "..agent[key])
				end
			end
		end,
		dumpGenes = function(self, args)
			agent["Rewards"] = {}
			agent["Punishments"] = {}
			agent.Past = {}
			local data = "local "..NN:tostring(NN:deepCopy(agent),"agent").."return agent"
			if not love.filesystem.getInfo( args.."/"..agent.GenePool, "directory" ) then
				love.filesystem.createDirectory( args.."/"..agent.GenePool )
			end
			love.filesystem.write( args.."/"..agent.GenePool.."/"..agent.id, data )
		end,
		dumpData = function(self, args)
			local tdata = {}
			tdata.Rewards = agent["Rewards"]
			tdata.Punishments = agent["Punishments"]
			local data = "local "..NN:tostring(NN:deepCopy(tdata),"data").."return data"
			if not love.filesystem.getInfo( args.."/"..agent.DataPool, "directory" ) then
				love.filesystem.createDirectory( args.."/"..agent.DataPool )
			end
			local suffix = #love.filesystem.getDirectoryItems( args.."/"..agent.DataPool )+1
			while love.filesystem.getInfo( args.."/"..agent.DataPool.."/"..agent.id..tostring(suffix), "file" ) do suffix = suffix+1 end
			love.filesystem.write( args.."/"..agent.DataPool.."/"..agent.id..tostring(suffix), data )
		end,
		splice = function(self, args)
			local files = love.filesystem.getDirectoryItems( args.."/"..agent.GenePool )
			local selection = {}
			for i,file in ipairs(files) do
				local ch = love.filesystem.load(args.."/"..agent.GenePool.."/"..file)
				selection[i] = ch()
			end
			local sorted = {}
			local size = #selection
			for i=1, size do
	
				local spot, topfit = 1, 0
				for s,a in ipairs(selection) do
					local afit = a.RPoints/a.PPoints
					if afit >= topfit then
						spot, topfit = s, afit
					end
				end
				sorted[i] = NN:deepCopy(selection[spot])
				table.remove(selection,spot)
				if i/size >= agent.SelectionPercent or #selection == 0 then
					break
				end
			end
			local tempID = agent["id"]
			agent = sorted[math.random(1,#sorted)]
			agent["id"] = tempID
			local cr = CrossoverOptions[agent.CrossoverType]
			cr(agent,sorted)
			for l,layer in ipairs(agent.Model) do
				for n,neuron in ipairs(layer.Net) do
					for w,weight in ipairs(neuron.W) do
						if   math.random() <= agent.MutationRate  then
							weight = weight+(((math.random()-0.5)*2)*agent.MutationIntensity)
							
						end
					end
					if math.random() <= agent.MutationRate then
						neuron.B = neuron.B+(((math.random()-0.5)*2)*agent.MutationIntensity)
					end
				end
			end
		end,
		updateParams = function(self, args)
			for key,item in pairs(args) do
				if type(item) == "table" then
					agent[key] = NN:deepCopy(item)
				else
					agent[key] = item
				end
			end
		end,
		kill = function(self, args)
			Running = false
		end,
		send = function(self, data)
			
			local target = false
			if type(data[#data]) == "string" then
				updateStatus({status = "Recording", data = #agent["Punishments"]+#agent["Rewards"]})
				target = NN:deepCopy(love.thread.getChannel( agent["id"].."target" ):demand())
				data[#data] = nil
			else
				updateStatus({status = "Predicting",Fitness = agent.Fitness})
			end
			if target then
				if not next(agent.Action) then
					local startTime = love.timer.getTime()
					agent.Action = NN:predict(agent.Model,data)
					agent.ModelPredictTiming = love.timer.getTime() - startTime
				else
					love.timer.sleep(agent.ModelPredictTiming)
				end
			else
				agent.Action = NN:predict(agent.Model,data)
			end
			
			agent.Time = agent.Time+1
			if agent.Random then
				local rng = RandomOptions[agent.Random]
				if rng then
					rng(agent)
				else
					error(tostring(agent.Random).." - is not an option for the Random setting")
				end
			end
			if agent.Record then
				if not agent["Past"] then
					agent["Past"] = {}
				end
				local d = {}
				d.T = target or NN:deepCopy(agent.Action)
				d.I = NN:deepCopy(data)
				table.insert(agent.Past,d)
			end
			if agent.MemoryLimit then
				if agent.Time >= agent.MemoryLimit then
					agent.Past = {}
					agent.Time = 0
					NN:forget(agent.Model)
				end
			end
			love.thread.getChannel( agent["id"].."received" ):push(agent.Action)
		end,
	}
	
	
	
	
	while Running do
		--updateStatus({status = "idle",completed = 0})
		command = love.thread.getChannel( agent["id"].."sent" ):demand()
		--updateStatus({status = "Processing",completed = 0})
		if type(command[1]) == "string" then
			if command[1] ~= "reward" and command[1] ~= "punish" and command[1] ~= "forget" then
			--print(command[1])
			end
			Commands[command[1]](Commands, command[2])
		else
			Commands.send(Commands, command)
		end
	end


else


	if love then
		local drawArc = {
			FF = {
				newdrawLayer = function(layer,drawmodel,model)
					local drawLayer = {}
					drawLayer.poses = {}
					local scalerx,scalery = 1/#model ,1/#layer.Net
					
					
					for n,neuron in ipairs(layer.Net) do
						local lineTab = {}
						for w,weight in ipairs(neuron.W) do
							
						end
						drawLayer.poses[n] = {X = (#drawmodel+1)*scalerx,Y = n*scalery}
						
						
					end
					
					return drawLayer
				end,
			},
		}
		
		love.NN = {
			makeDrawable = function(model)
				local drawmodel = {}
				for l,layer in ipairs(model) do
					local dArc = drawArc[layer.Arc]
					table.insert(drawmodel,dArc.newdrawLayer(layer,drawmodel,model))
					
				end
				return drawmodel
			end,
		}
	end
	local pathToThisFile = (...):gsub("%.", "/") .. ".lua"
	local Agent = {
		_VERSION     = 'Agent v1.0.0',
		_DESCRIPTION = 'AI for LÖVE',
		_URL         = 'https://github.com/PhytoEpidemic/love2d-Agents',
		_LICENSE     = [[
MIT LICENSE

Copyright (c) 2024 PhytoEpidemic

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ]],
	}
	function Agent.new(params)
		params = params or {}
		local Thread = love.thread.newThread(pathToThisFile)
		Thread:start({
			Model = params["Model"],
			Layers = params["Layers"] or false,
			Seed = params["Seed"] or os.time(),
			oscillator = params["oscillator"],
			GenePool = params["GenePool"] or "default",
			DataPool = params["DataPool"] or false,
			SelectionPercent = params["SelectionPercent"] or 0.7,
			MutationRate = params["MutationRate"] or 0.3,
			MutationIntensity = params["MutationIntensity"] or 0.1,
			CrossoverRate = params["CrossoverRate"] or 0.1,
			CrossoverType = params["CrossoverType"] or "two_point",
			CrossoverIntensity = params["CrossoverIntesity"] or 0.3,
			Action = {},
			id = params["id"] or "0",
			Past = {},
			Time = params["Time"] or 0,
			Record = params["Record"] or false,
			Random = params["Random"] or false,
			Threshold = params["Threshold"] or 0,
			DecayThreshold = params["DecayThreshold"] or 0,
			MemoryLimit = params["MemoryLimit"] or false,
			Training = false,
			RPoints = 0,
			PPoints = 0,
			Opt = params["Model"]["Opt"] or "batch",
			Fitness = 0,
			TargetUpperLimit = 1,
			TargetLowerLimit = 0,
		})
		--print(params["id"])
		local self = {
			Action = {},
			id = params["id"] or "0",
			Training = false,
			RPoints = 0,
			PPoints = 0,
		}
		for k,v in pairs(Agent) do
			if k ~= "new" then
				self[k] = v
			end
		end
		
		self.__index = self
		love.thread.getChannel( self.id.."status" ):push({
			status = "idle",completed = 0
		})
		return self
	end
	
	function Agent:initGenePool(path,new)
		if path then
			if not love.filesystem.getInfo( path, "directory" ) or new then
				love.filesystem.createDirectory( path )
			end
			Agent.GenePoolPath = path
		end
		
	end
	function Agent:dumpGenes()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"dumpGenes",Agent.GenePoolPath})
	end
	function Agent:initDataPool(path, new, compress)
		if path then
			if not love.filesystem.getInfo( path, "directory" ) or new then
				love.filesystem.createDirectory( path )
			end
			if compress then
				io.popen([[compact /C /S:"]]..(love.filesystem.getSaveDirectory())..[[/]]..path..[[" 2>&1]]):close()
			end
			Agent.DataPoolPath = path
		end
		
	end
	function Agent:dumpData()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"dumpData",Agent.DataPoolPath})
	end
	function Agent:splice()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"splice",Agent.GenePoolPath})
	end
	function Agent:updateParams(params)
		if type(params) ~= "table" then
			error("Must be a table for both arguments: arg1 = "..type(params))
		end
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"reward",params})
	end
	function Agent:forget()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"forget"})
	end
	function Agent:kill()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"kill"})
	end
	function Agent:send(data,target)
		if type(data) ~= "table" then
			error("Must be a table for both arguments: arg1 = "..type(data))
		end
		
		local channelR = love.thread.getChannel( self.id.."received" )
		local resultC = channelR:getCount( )
		local predictedAction = false
		if resultC > 0 then
			self.Training = false
			self.Action = channelR:pop()
			predictedAction = self.Action
			return predictedAction
		end
		local channelS = love.thread.getChannel( self.id.."sent" )
		if target then
			data[#data+1] = "target"
		end
		if channelS:getCount( ) > 0 then
			local temp = channelS:peek()
			if temp then
				if type(temp[1]) == "number" then

				end
			end
		else
			if target then
				love.thread.getChannel( self.id.."target" ):push(target)
			end
			channelS:push(data)
		end
		
	end
	function Agent:reward(points)
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"reward",points or 0})
		self.RPoints = self.RPoints+points or 0
	end
	function Agent:save(path)
		--love.filesystem.createDirectory(path)
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"save",path})
	end
	function Agent:load(path)
		--love.filesystem.createDirectory(path)
		local channelS = love.thread.getChannel( self.id.."sent" )
		local file = love.filesystem.load(path)
		if not file then
			return false
		end
		local Agent = file()
		for k,v in pairs(Agent) do
			if k ~= "new" then
				self[k] = v
			end
		end
		channelS:push({"load",path})
	end
	function Agent:punish(points)
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"punish",points or 0})
		self.PPoints = self.PPoints+points or 0
	end
	function Agent:resetPoints()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"resetPoints"})
		self["PPoints"] = 0
		self["RPoints"] = 0
	end
	function Agent:train(epochs)
		local channelS = love.thread.getChannel( self.id.."sent" )
		epochs = epochs or 1
		for e=1,epochs do
			print("Epochs",e)
			channelS:push({"train",Agent.DataPoolPath})
		end
		self.Training = true
		local channelR = love.thread.getChannel( self.id.."received" )
		channelR:clear()
	end
	function Agent:decay(params)
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"decay",params})
	end
	function Agent:removeData()
		local channelS = love.thread.getChannel( self.id.."sent" )
		channelS:push({"removeData"})
	end
	function Agent:getStatus()
		return love.thread.getChannel( self.id.."status" ):peek()
	end
	
	return Agent

end






