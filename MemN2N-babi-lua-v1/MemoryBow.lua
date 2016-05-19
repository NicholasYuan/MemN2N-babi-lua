-- MemoryBow

function MemoryBow( config, hid, story )
	local memory = nil
	local sharelist = {}

	local Aintmp = nn.LookupTable(config.voc_sz, config.input_dim, config.max_grad_norm)(story)
	local Aout = nn.Sum(3)(Ain)
	table.insert(sharelist[2], Ain)

	local Cintmp = nn.LookupTable(config.voc_sz, config.out_dim, config.max_grad_norm)(story)
	local Cout = nn.Sum(3)(Cin)
	table.insert(sharelist[3], Cin)

	local x = nn.MatVecProb(true)({Aout,hid})
	local xout = nn.SoftMaxNB(false)(x)

	local Out = nn.MatVecProb(false)({Cout,xout})
	memory = Out
	return memory, sharelist
end