
require('paths')
package.path = package.path .. ';/home/exp/program/MenNN/MemN2N-babi-lua/?.lua;'

require('lfs')
require('nn')
require('nngraph')
-- require('cunn')
local tds = require('tds')
dofile('model.lua')
dofile('data.lua')
dofile('config_babi.lua')
dofile('util.lua')
dofile('checkGrad.lua')

dofile('MatVecProb.lua')
dofile('CrossEntropyLossNB.lua')
dofile('EleMult.lua')
dofile('LookUpTableNB.lua')
dofile('LinearNB.lua')
dofile('MemoryBow.lua')
dofile('MemoryL.lua')
dofile('SoftMaxNB.lua')


base_dir = '/home/exp/program/MemNN/tasks_1-20_v1-2/en'
t = 1

fpath = 'qa'..t

-- data_path = fpath..'_*_train.txt'
-- test_data_path = fpath..'_*_test.txt'
data_path = {}
test_data_path = {}
for file in lfs.dir(base_dir) do
	if(string.match(file, fpath..'_.+_train.txt') ~= nil) then
		table.insert(data_path,base_dir..'/'..file)
	end
	if(string.match(file, fpath..'_.+_test.txt') ~= nil) then
		table.insert(test_data_path,base_dir..'/'..file)
	end
end
print(unpack(data_path))
print(unpack(test_data_path))


dict = tds.hash()
dict['nil'] = 1



story, questions, qstory, dict = paraseBabiTask(data_path, dict, false)
test_story, test_questions, test_qstory, dict = paraseBabiTask(test_data_path, dict, false)
config = build_config(story, dict, questions)
config.lrate = config.init_lrate


train_range = math.floor(0.9 * questions:size(1))
val_range = questions:size(1)-(math.floor(0.9 * questions:size(1))+1)

model = {}
model.memnn,shareList = build_model(config, dict)
model_param, model_paramdx = model.memnn:getParameters()
model_param:normal(0,0.1)
model.criterion = nn.CrossEntropyLossNB(false,true)
-- train
local function train( )
	local N = math.floor(train_range/config.bsz)
	local cost_total = 0
	local err_total = 0
	local y = nil

	for k=1,N do
		local batch = torch.randperm(train_range):narrow(1,1,config.bsz):clone()
		batch = torch.LongTensor(batch:long())
		local offset = torch.zeros(config.bsz)
		local in_answer = questions:t()[3]:index(1,batch)
		local in_query = torch.zeros(config.bsz,story:size(3))
		local in_story = torch.zeros(config.bsz, config.sz, config.max_words)
		in_story[{{},{},{}}] = dict['nil']
		for b=1,config.bsz do
			d = story[{questions[{batch[b],1}],{1,questions[{batch[b],2}]},{}}]
			offset[b] = math.max(0, d:size(1)-config.sz) 

			d = d[{{1+offset[b],-1},{}}]
			if config.enable_time then
				if config.randomize_time > 0 then
					local nblank = math.random(0,math.ceil(d:size(1) * config.randomize_time))
					local rt = torch.randperm(d:size(1) + nblank)
					rt[torch.gt(rt,config.sz)] = config.sz
					local rtm,_ = torch.sort(rt[{{1,d:size(1)}}],1,true) + #dict
					d = torch.cat(d,rtm:resize(rtm:size(1),1),2)
				else
					d = torch.cat(d,(torch.range(d:size(1),1,-1) + #dict):view(-1,1),2)
				end
			end
			in_story[{b,{1,d:size(1)},{1,d:size(2)}}] = d
			in_query[{b,{}}] = qstory[{batch[b],{}}]
		end
		local x = { in_query, in_story}


		local pred = model.memnn:forward(x)
		local class,class_index = torch.max(pred,2)
		class_index = class_index:squeeze():long()
		local cost = model.criterion:forward({pred, in_answer})
		-- print("cost",cost,config.lrate)
		local err = torch.ne(class_index, in_answer:long()):sum()

		cost_total = cost_total + cost
		err_total = err_total + err
		model_paramdx:zero()
		y = model.criterion:backward(pred,in_answer)
		model.memnn:backward(x,y)

		local gn = model_paramdx:norm()
		if gn > config.max_grad_norm * 2.8 then
			model_paramdx = model_paramdx:mul(config.max_grad_norm / gn)
		end
		model.memnn:updateParameters(config.lrate)

		for i=1,config.nhops do
			shareList[2][i][1].data.module.weight[{{},1}]=0
			shareList[3][i][1].data.module.weight[{{},1}]=0
		end

	end
	return cost_total/N/config.bsz, err_total
end

local function val(	 )
	local N = math.floor(val_range/config.bsz)
	local cost_total = 0
	local err_total = 0
	local y = torch.ones(1)
	
	for k=1,N do
		local batch = (torch.range(1,config.bsz) + (k-1)*config.bsz) + train_range
		batch = torch.LongTensor(batch:long())
		local in_answer = questions:t()[3]:index(1,batch)
		local in_query = torch.zeros(config.bsz,story:size(3))
		local in_story = torch.zeros(config.bsz, config.sz, config.max_words)
		in_story[{{},{},{}}] = dict['nil']
		for b=1,config.bsz do
			d = story[{questions[{batch[b],1}],{1,questions[{batch[b],2}]},{}}]
			d = d[{{math.max(1,d:size(1)-config.sz+1),-1},{}}]
			if config.enable_time then
				d = torch.cat(d,(torch.range(d:size(1),1,-1) + #dict):view(-1,1),2)
			end
			in_story[{b,{1,d:size(1)},{1,d:size(2)}}] = d
			in_query[{b,{}}] = qstory[{batch[b],{}}]
		end
		local x = {in_query, in_story}
		local pred = model.memnn:forward(x)
		local cost = model.criterion:forward({pred, in_answer})
		local class,class_index = torch.max(pred,2)
		class_index = class_index:squeeze():long()
		local err = torch.eq(class_index, in_answer:long()):sum()

		cost_total = cost_total + cost
		err_total = err_total + class_index:size(1) - err

	end
	return cost_total/N/config.bsz, err_total
end 

local function test(  )
	local N = math.floor(test_questions:size(2)/config.batch_size)
	local cost_total = 0
	local err_total = 0
	local y = torch.ones(1)
	

	for k=1,N do
		local batch = (torch.range(1,config.bsz) + (k-1)*config.bsz)
		batch = torch.LongTensor(batch:long())
		local in_query = torch.zeros(config.bsz, test_story:size(3))
		local in_story = torch.zeros(config.bsz, config.sz, config.max_words)
		in_query[{{},{}}] = dict['nil']
		in_story[{{},{},{}}] = dict['nil']
		local in_answer = test_questions:t()[3]:index(1,batch)
		for b=1,config.bsz do
			d = test_story[{test_questions[{batch[b],1}],{1,test_questions[{batch[b],2}]},{}}]
			d = d[{{math.max(1,-1-config.sz+1),-1},{}}]
			if config.enable_time then
				d = torch.cat(d,(torch.range(d:size(1),1,-1) + #dict):view(-1,1),2)
			end
			in_story[{b,{1,d:size(1)},{1,d:size(2)}}] = d
			in_query[{b,{}}] = test_qstory[{batch[b],{}}]
		end
		local x = {in_query, in_story}

		local pred = model.memnn:forward(x)
		local class,class_index = torch.max(pred,2)
		class_index = class_index:squeeze():long()
		local cost = model.criterion:forward({pred, in_answer})
		local err = torch.eq(class_index, in_answer:long()):sum()

		cost_total = cost_total + cost
		err_total = err_total + class_index:size(1) - err
	end
	return cost_total/N/config.bsz, err_total
end

local total_err = 0
local total_cost = 0
local total_num = 0
local loss0 = nil
for ep=1,config.nepochs do
	
	local timer = torch.Timer()

	local train_cost,err_train = train()
	local val_cost,err_val = val()
	local test_cost,err_test = test()

	local time = timer:time().real

	if ep % config.lrate_decay_step == 0 then
		config.lrate = config.lrate * 0.5
	end

	if ep % config.print_every == 0 then
		print(string.format("%d/%d (epoch %.3f), train_cost = %6.8f, val_cost = %6.8f, test_cost = %6.8f,\ngrad/param norm = %6.4e, time/batch = %.4fs",
			ep,config.nepochs,ep/config.nepochs,
			train_cost,
			val_cost,test_cost,
			model_paramdx:norm()/model_param:norm(), time))
		print(string.format("train_err = %6.f %6.8f, val_err = %6.f %6.8f, test_err = %6.f %6.8f"
			,err_train, err_train/train_range, err_val,err_val/val_range
			, err_test, err_test/test_questions:size(2)))
	end

	if ep % 20 == 0 then collectgarbage() end

	if train_cost ~= train_cost then
		print(train_cost)
		print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end

    -- if loss0 == nil then loss0 = loss[1] end
    -- if loss[1] > loss0 * 3 then
    --     print('loss is exploding, aborting.')
    --     print (loss[1])
    --     break -- halt
    -- end

end


