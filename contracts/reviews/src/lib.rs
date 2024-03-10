use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    pubkey::Pubkey,
};

entrypoint!(process_instruction);

fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();

    // Get the account that holds the reviews
    let reviews_account = next_account_info(accounts_iter)?;

    // Deserialization of the stored data
    let mut reviews_data = if reviews_account.data.borrow().is_empty() {
        vec![]
    } else {
        match Vec::<String>::try_from_slice(&reviews_account.data.borrow()) {
            Ok(data) => data,
            Err(..) => return Err(ProgramError::InvalidAccountData),
        }
    };

    // Assume instruction_data is a UTF-8 encoded string representing the new review hash
    let new_review_hash = std::str::from_utf8(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?
        .to_string();

    // Append the new review
    reviews_data.push(new_review_hash);

    // Serialize and save the updated reviews back into the account
    reviews_data.serialize(&mut &mut reviews_account.data.borrow_mut()[..])?;

    Ok(())
}
