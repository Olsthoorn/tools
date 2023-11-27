classdef fdmPckg
    properties
        name
        time
        size
    end
    methods
        function o = fdmPckg(lbl,gr,t,data)
            o.name = lbl;
            o.size = gr.size;
            o.time = t;
        end
    end
end