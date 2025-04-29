`timescale 1ns / 1ps
// 实验要求
    // 补全模块（阶段三）

module CSR_Regfile(
    input wire clk,
    input wire rst,
    input wire CSR_write_en,
    input wire [11:0] CSR_write_addr,
    input wire [11:0] CSR_read_addr,
    input wire [31:0] CSR_data_write,
    output wire [31:0] CSR_data_read
    );
    
    // TODO: Complete this module

    /* FIXME: Write your code here... */
    reg [31:0] reg_file[12'hfff:0];
    integer i;

    // init register file
    initial
    begin
        for(i = 1; i <= 32'hfff; i = i + 1) 
            reg_file[i][31:0] <= 32'b0;
    end

    always@(posedge clk or posedge rst)
    begin 
        if (rst)
            for (i = 1; i <= 32'hfff; i = i + 1) 
                reg_file[i][31:0] <= 32'b0;
        else if(CSR_write_en )
            reg_file[CSR_write_addr] <= CSR_data_write;   
    end

    // read data changes when address changes
    assign CSR_data_read = reg_file[CSR_read_addr];

endmodule
