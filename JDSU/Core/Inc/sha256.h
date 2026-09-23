#ifndef __SHA256_H
#define __SHA256_H

#include <stdint.h>

typedef struct
{
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t block[64];
    uint32_t block_length;
} Sha256Context;

void Sha256_Init(Sha256Context *context);
void Sha256_Update(Sha256Context *context, const uint8_t *data, uint32_t length);
void Sha256_Final(Sha256Context *context, uint8_t digest[32]);
void HmacSha256(const uint8_t *key, uint32_t key_length,
                const uint8_t *data, uint32_t data_length,
                uint8_t digest[32]);

#endif
