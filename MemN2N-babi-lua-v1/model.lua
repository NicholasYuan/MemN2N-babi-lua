
function build_model(config, dict)
    local query = nn.Identity()()
    local story = nn.Identity()()

    local sharelist = {}

    -- for query
    local uin = nn.LookUpTableNB(config.voc_sz, config.input_dim, config.max_grad_norm)(query)
    sharelist[1] = uin
    sharelist[2] = {}
    sharelist[3] = {}
    local Muout = nil
    if config.use_bow == false then
        local Muouttmp = nil
        if config.enable_time then
            Muouttmp = nn.EleMult(config.weight[{{1,-2},{}}])(uin)
        else 
            Muouttmp = nn.EleMult(config.weight)(uin)
        end
        Muout = nn.Sum(2)(Muouttmp)
    else
        Muout = nn.Sum(2)(uin)
    end
    local uout = Muout


    -- for loop memory
    local hid = {}
    hid[0] = uout
    local memory = {}
    local proj = {}
    for i=1,config.nhops do
        -- local P1 = nn.ParallelTable()
        local P1 = nil
        local sharell = {}
        if config.use_bow then
            memory[i], sharell = MemoryBow(config, hid[i-1], story)
        else
            memory[i], sharell = MemoryL(config, hid[i-1], story)
        end
        table.insert(sharelist[2], sharell[2])
        table.insert(sharelist[3], sharell[3])
        -- P1:add(memory[i])
        if config.add_proj then
            proj[i] = nn.LinearNB(config.input_dim, config.input_dim, config.max_grad_norm)
                (nn.Transpose(1,1)(hid[i-1]))
            P1 = proj[i]
        else
            P1 = hid[i-1]
        end
        local D = nn.CAddTable()({memory[i], P1})
        if config.add_nonlin then
            hid[i] = nn.ReLU()(D)
        else
            hid[i] = D
        end
    end
    
    local Out = nn.LinearNB(config.out_dim, config.voc_sz, true, config.max_grad_norm)(hid[#hid])
    local pred = nn.SoftMaxNB(true)(Out)

    local model = nn.gModule({ query, story}, {pred})
    sharelist[4] = Out
    -- model:cuda()

    -- do share
    if config.share_type == 1 then
        local m1 = sharelist[1].data.module
        local m21 = sharelist[2][1][1].data.module
        m21:share(m1,'weight')

        for i=2,config.nhops do
            local m2i = sharelist[2][i][1].data.module
            local m3i_1 = sharelist[3][i-1][1].data.module
            m2i:share(m3i_1,'weight')
        end
        local m4 = sharelist[4].data.module
        local m3i = sharelist[3][config.nhops][1].data.module
        m4:share(m3i,'weight')
    elseif config.share_tpye == 2 then
        local m2 = sharelist[2][1][1].data.module
        local m3 = sharelist[3][1][1].data.module
        for i=2,config.nhops do
            local m2i = sharelist[2][i][1].data.module
            local m3i = sharelist[3][i][1].data.module
            m2i:share(m2,'weight')
            m3i:share(m3,'weight')
        end
    end

    if config.add_proj then
        local proj1 = proj[1].data.module
        for i=2,config.nhops do
            local proji = proj[i].data.module
            proji:share(proj1,'weight')
        end
    end

    return model, sharelist
end
