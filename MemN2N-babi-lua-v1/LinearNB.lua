
local LinearNB, parent = torch.class('nn.LinearNB', 'nn.Module')

function LinearNB:__init(inputSize, outputSize, do_transpose, max_grad_norm)
   parent.__init(self)

   self.outputSize = outputSize
   self.inputSize = inputSize
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.max_grad_norm = max_grad_norm
   self:reset()
end

function LinearNB:reset(stdv)
   self.weight = torch.randn(self.weight:size())*0.1
end

function LinearNB:updateOutput(input)
   if input:dim() == 1 then
       self.output:resize(self.outputSize)
       self.output:zero()
       self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
       local nframe = input:size(1)
       local nunit = self.outputSize
       self.output:resize(nframe, nunit):zero()
       if not self.addBuffer or self.addBuffer:size(1) ~= nframe then
           self.addBuffer = input.new(nframe):fill(1)
       end
       if nunit == 1 then
           -- Special case to fix output size of 1 bug:
           self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
       else
           self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNB:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
          self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
          self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function LinearNB:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      local nunit = self.outputSize
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.gradWeight:select(1,1):addmv(scale,
                                           input:t(), gradOutput:select(2,1))
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
      end
   end
   local norm = self.gradWeight:norm()
   if norm > self.max_grad_norm then
      self.gradWeight = self.gradWeight:mul(self.max_grad_norm/norm)
   end
end
