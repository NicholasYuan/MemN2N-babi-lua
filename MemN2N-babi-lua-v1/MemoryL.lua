-- MemoryBow

function MemoryL( config, hid, story )
	local memory = nil
	local sharelist = {}
	sharelist[2] = {}
	sharelist[3] = {}

	local Ain = nn.LookUpTableNB(config.voc_sz, config.input_dim, config.max_grad_norm)(story)

	local Aout = nn.Sum(3)(nn.EleMult(config.weight)(Ain))
	
	table.insert(sharelist[2], Ain)

	local Cin = nn.LookUpTableNB(config.voc_sz, config.out_dim, config.max_grad_norm)(story)

	local Cout = nn.Sum(3)(nn.EleMult(config.weight)(Cin))

	table.insert(sharelist[3], Cin)

	local x = nn.MatVecProb(false)({Aout,hid})
	local xout = nn.SoftMaxNB(false)(x)

	local Out = nn.MatVecProb(true)({Cout,xout})
	memory = Out
	return memory, sharelist
end