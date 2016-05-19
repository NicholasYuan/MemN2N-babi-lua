-- SoftMaxNB

local SoftMaxNB, parent = torch.class("nn.SoftMaxNB", "nn.Module")

function SoftMaxNB:__init( bpTurn )
	self.bpTurn = bpTurn
end

function SoftMaxNB:updateOutput( input )
	local maxinput,_ = torch.max(input,2)
	maxinput = maxinput:squeeze()

	for i=1,maxinput:size(1) do
		input[i]:csub(maxinput[i])
	end
	input = input + 1
	local a = torch.exp(input)
	self.output = torch.zeros(a:size()):type(input:type())
	asum = torch.sum(a,2):squeeze()
	for i=1,a:size(1) do
		self.output[i] = a[i]:div(asum[i])
	end
	return self.output
end

function SoftMaxNB:updateGradInput( input, gradOutput )
	if self.bpTurn == false then
		local ztmp = torch.sum(torch.cmul(self.output, gradOutput),2):squeeze()
		local z = torch.zeros(gradOutput:size()):type(gradOutput:type()):copy(gradOutput)
		for i=1,ztmp:size(1) do
			z[i]:csub(ztmp[i])
		end
		self.gradInput = torch.cmul(self.output, z)
	else
		self.gradInput = gradOutput
	end
	return self.gradInput
end