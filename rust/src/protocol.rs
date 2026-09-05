use crate::pack::ConnectomePack;

pub const RIGHT_SUGAR_GRN_IDS: [u64; 21] = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
];

pub const DNA02_RIGHT_ID: u64 = 720575940604737708;
pub const DNA02_LEFT_ID: u64 = 720575940629327659;
pub const MN9_LEFT_ID: u64 = 720575940660219265;

pub fn sugar_indices(connectome: &ConnectomePack) -> (Vec<u32>, Vec<u64>) {
    let mut indices = Vec::new();
    let mut missing = Vec::new();
    for flywire_id in RIGHT_SUGAR_GRN_IDS {
        match connectome
            .neuron_ids
            .iter()
            .position(|candidate| *candidate == flywire_id)
        {
            Some(index) => indices.push(index as u32),
            None => missing.push(flywire_id),
        }
    }
    (indices, missing)
}

#[cfg(test)]
mod tests {
    use super::{RIGHT_SUGAR_GRN_IDS, sugar_indices};
    use crate::pack::ConnectomePack;

    #[test]
    fn maps_available_ids_and_reports_missing_ids() {
        let connectome = ConnectomePack::from_arrays(
            [RIGHT_SUGAR_GRN_IDS[1], 42, RIGHT_SUGAR_GRN_IDS[0]],
            [0, 0, 0, 0],
            [],
            [],
        )
        .unwrap();

        let (indices, missing) = sugar_indices(&connectome);

        assert_eq!(indices, [2, 0]);
        assert_eq!(missing.len(), RIGHT_SUGAR_GRN_IDS.len() - 2);
    }
}
