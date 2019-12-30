/* Hello World Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "app_priv.h"
//Entry point of our hello_world_tensorflow component
#include "main_functions.h"


void app_main()
{
    //int i = 0;
    //app_driver_init();
    printf("\nCalling setup from main\n");
    setup();
    printf("\nReturned from setup to main/n");
    while (1) {
        //printf("[%d] Hello world!\n", i);
        printf("\nCalling loop from main\n");
        loop();
        printf("\nReturned from loop to main\n");
        //vTaskDelay(5000 / portTICK_PERIOD_MS);
    }
}


