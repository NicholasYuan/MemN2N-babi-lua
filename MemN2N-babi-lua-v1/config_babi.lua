
function build_config( story, dict, questions )
	local cmd = torch.CmdLine()
	cmd:option('--gpu', 1, 'GPU id to use')
	cmd:option('--print_every', 1, 'print every times')
	cmd:option('--batch_size', 32)
	cmd:option('--nhops', 3, 'number of hops')
	cmd:option('--nepochs', 1000)
	cmd:option('--lrate_decay_step', 25, 'reduce learning rate by half every n epochs')
	cmd:option('--enable_time',true,'add time embedings')
	cmd:option('--use_bow', false,'use Bag-of-Words instead of Position-Encoding')
	cmd:option('--share_type', 1,'1: adjecent, 2: layer-wise weight tying')
	cmd:option('--randomize_time', 0.1,'amount of noise injected into time index')
	cmd:option('--add_proj', false,'add linear layer between internal states')
	cmd:option('--add_nonlin', false,'add non-linear layer to internal states')

	cmd:option('--init_lrate', 0.01,'initial learning rate')
	cmd:option('--max_grad_norm', 40)
	cmd:option('--input_dim', 20)
	cmd:option('--out_dim', 20)

	cmd:option('--linear_start', false)


	config = cmd:parse(arg or {})
	if config.linear_start then
		config.ls_nepochs = 20
		config.ls_lrate_decay_step = 21
		config.ls_init_lrate = 0.01/2
		config.init_lrate = 0.01/2
	end

	config.sz = math.min(50, story:size(2))
	config.voc_sz = #dict
	config.bsz = config.batch_size
	config.max_words = story:size(3)
	if config.enable_time then
		config.voc_sz = config.voc_sz + config.sz
		config.max_words = config.max_words + 1
	end

	if config.use_bow == false then
        config.weight = torch.ones(config.input_dim, config.max_words)
        for i=1,config.input_dim do
            for j=1,config.max_words do
                config.weight[{i,j}] = (i-(config.input_dim+1)/2) * (j-(config.max_words +1)/2)
            end
        end
        config.weight = config.weight * 4 / config.input_dim / config.max_words + 1
        config.weight = config.weight:t()
        -- config.weight:cuda()
    end

	return config
end